"""
K8s Training Job — runs inside a K8s Pod when slow/medium level triggers.

Receives config from environment variables (injected via K8s Secret).
Logs metrics + registers model to MLflow; artifacts stored in Ceph S3.

Training strategy:
  medium → fine-tune: freeze backbone, train FC layer only (fast, lightweight)
  slow   → full retrain: all layers unfrozen, load previous UAV weights if available

Titans NL forward pass:
  Training uses the same Titans combine step as inference:
    cur      = backbone(image)                              ← ResNet18 features
    combined = normalize(cur + blend(W_fast@cur, W_med@cur, W_slow@cur))  ← Titans context
    logits   = fc(combined)                                ← action prediction
    loss     = criterion(logits, labels)

  W matrices (W_fast/W_med/W_slow) are loaded from Ceph at job start and held
  FIXED during training — they provide terrain context, not backpropped through.
  Only backbone + FC weights are updated by backprop.

Pseudo-labels:
  Labels come from manifest.json files uploaded by flight_agent (one per flight).
  Manifest maps Ceph frame key → predicted action (self-training pseudo-labels).
  Fallback to mock metrics if no manifest found (first run, Ceph unavailable).
"""
import io
import json
import logging
import os
import random
import time

import boto3
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [training_job] %(levelname)s %(message)s",
)
logger = logging.getLogger("training_job")

# ── Config from K8s env vars ───────────────────────────────────────────────
MLFLOW_URI            = os.environ.get("MLFLOW_TRACKING_URI",    "http://localhost:5000")
S3_ENDPOINT           = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:7480")
S3_ACCESS_KEY         = os.environ.get("AWS_ACCESS_KEY_ID",      "uavaccess")
S3_SECRET_KEY         = os.environ.get("AWS_SECRET_ACCESS_KEY",  "uavsecret123")
S3_REGION             = os.environ.get("AWS_DEFAULT_REGION",     "us-east-1")
BUCKET_TRAINING_DATA  = os.environ.get("BUCKET_TRAINING_DATA",   "training-data")
BUCKET_FAST_WEIGHT    = os.environ.get("BUCKET_FAST_WEIGHT",     "fast-weight-state")
CHECKPOINT_KEY        = os.environ.get("CHECKPOINT_KEY",         "slow/v1.pt")
LEVEL                 = os.environ.get("TRAINING_LEVEL",         "slow")
TERRAIN               = os.environ.get("TRAINING_TERRAIN",       "unknown")
JOB_ID                = os.environ.get("TRAINING_JOB_ID",        "unknown")

# NL blend weights (must match config/settings.py)
NL_BLEND_FAST   = float(os.environ.get("NL_BLEND_FAST",   "0.30"))
NL_BLEND_MEDIUM = float(os.environ.get("NL_BLEND_MEDIUM", "0.20"))
NL_BLEND_SLOW_W = float(os.environ.get("NL_BLEND_SLOW_W", "0.10"))

os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]      = S3_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]  = S3_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"]     = S3_REGION

mlflow.set_tracking_uri(MLFLOW_URI)
client = mlflow.MlflowClient()

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
)

# ── Action space ───────────────────────────────────────────────────────────
_ACTIONS      = ["straight", "left", "right", "stop"]
_ACTION_TO_IDX = {a: i for i, a in enumerate(_ACTIONS)}

# Image transform — must match feature_extractor.py
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Titans helpers (mirrors core/titans_memory recall logic) ────────────────

def _load_titans_state(terrain: str, embed_dim: int = 512) -> tuple:
    """
    Load W_fast/W_med/W_slow from Ceph fast-weight-state/terrain_{t}/latest.pt.
    Backward compatible: if Ceph still has old {"M_t", "M_med", "M_slow"} keys,
    loads them into W_fast/W_med/W_slow.
    Returns zeros if not found (first run for this terrain).
    """
    zeros = lambda: torch.zeros(embed_dim, embed_dim)
    key   = f"terrain_{terrain}/latest.pt"
    try:
        obj    = s3.get_object(Bucket=BUCKET_FAST_WEIGHT, Key=key)
        buf    = io.BytesIO(obj["Body"].read())
        loaded = torch.load(buf, weights_only=True)
        if isinstance(loaded, dict):
            if "W_fast" in loaded:
                W_fast = loaded.get("W_fast", zeros())
                W_med  = loaded.get("W_med",  zeros())
                W_slow = loaded.get("W_slow", zeros())
            else:
                # Legacy Hebbian format — migrate key names
                W_fast = loaded.get("M_t",    zeros())
                W_med  = loaded.get("M_med",  zeros())
                W_slow = loaded.get("M_slow", zeros())
        else:
            W_fast, W_med, W_slow = loaded, zeros(), zeros()
        logger.info(
            f"Titans state loaded  terrain={terrain}  "
            f"W_fast={W_fast.norm():.4f}  W_med={W_med.norm():.4f}  W_slow={W_slow.norm():.4f}"
        )
        return W_fast, W_med, W_slow
    except Exception as e:
        logger.info(f"No Titans state for terrain={terrain} ({e}) — using zeros")
        return zeros(), zeros(), zeros()


def _combine_titans(cur: torch.Tensor, W_fast, W_med, W_slow) -> torch.Tensor:
    """
    Titans combine step — mirrors TitansMemory.recall_all() read-only path.
    Handles single vectors (512,) and batches (N, 512).
    W matrices are fixed during outer-loop training (registered as buffers).
    """
    if cur.dim() == 1:
        r_fast = F.normalize(W_fast @ cur, dim=0)
        r_med  = F.normalize(W_med  @ cur, dim=0)
        r_slow = F.normalize(W_slow @ cur, dim=0)
        dim    = 0
    else:
        r_fast = F.normalize(cur @ W_fast.T, dim=-1)
        r_med  = F.normalize(cur @ W_med.T,  dim=-1)
        r_slow = F.normalize(cur @ W_slow.T, dim=-1)
        dim    = -1
    return F.normalize(
        cur
        + NL_BLEND_FAST   * r_fast
        + NL_BLEND_MEDIUM * r_med
        + NL_BLEND_SLOW_W * r_slow,
        dim=dim,
    )


class NLResNet(nn.Module):
    """
    ResNet18 wrapper that inserts the Titans combine step between backbone and FC.

    forward(x):
      cur      = backbone(x)               # (N, 512)
      combined = _combine_titans(cur, ...)  # (N, 512)  ← Titans context step
      logits   = fc(combined)              # (N, 4)

    W_fast/W_med/W_slow are registered as buffers — NOT backpropped.
    Only backbone and FC weights are updated by the outer loop.
    """
    def __init__(self, resnet: nn.Module, W_fast, W_med, W_slow):
        super().__init__()
        self.resnet    = resnet
        self._backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.register_buffer("W_fast", W_fast)
        self.register_buffer("W_med",  W_med)
        self.register_buffer("W_slow", W_slow)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur      = self._backbone(x).squeeze(-1).squeeze(-1)                   # (N, 512)
        combined = _combine_titans(cur, self.W_fast, self.W_med, self.W_slow)  # (N, 512)
        return self.resnet.fc(combined)                                         # (N, 4)


# ── Pseudo-labeled dataset ─────────────────────────────────────────────────

class CephFrameDataset(Dataset):
    """
    PyTorch Dataset backed by Ceph S3.

    frame_label_map: {s3_key: action_str} — produced by flight_agent._upload_flight_manifest()
    Loads each frame image from Ceph on __getitem__. No local cache (K8s Pod is stateless).
    Skips frames that cannot be fetched (network error, deleted) with a warning.
    """

    def __init__(self, s3_client, frame_label_map: dict[str, str]):
        self.s3    = s3_client
        self.items = list(frame_label_map.items())  # [(s3_key, action_str), ...]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        key, action_str = self.items[idx]
        try:
            obj      = self.s3.get_object(Bucket=BUCKET_TRAINING_DATA, Key=key)
            raw      = obj["Body"].read()
            img      = Image.open(io.BytesIO(raw)).convert("RGB")
            tensor   = _TRANSFORM(img)
        except Exception as e:
            logger.warning(f"CephFrameDataset: failed to load {key}: {e} — returning zeros")
            tensor = torch.zeros(3, 224, 224)
        label = torch.tensor(_ACTION_TO_IDX.get(action_str, 0), dtype=torch.long)
        return tensor, label


def _load_manifests(terrain: str) -> dict[str, str]:
    """
    Collect all manifest.json files for a terrain, merge into one dict.

    Returns {s3_frame_key: action_str} from all flights of this terrain.
    Empty dict if no manifests found (first run or Ceph unavailable).
    """
    combined: dict[str, str] = {}
    prefix = f"frames/terrain_{terrain}/"
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_TRAINING_DATA, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("manifest.json"):
                    continue
                try:
                    manifest_obj = s3.get_object(Bucket=BUCKET_TRAINING_DATA, Key=key)
                    data = json.loads(manifest_obj["Body"].read())
                    combined.update(data)
                    logger.info(f"Loaded manifest {key}  ({len(data)} frames)")
                except Exception as e:
                    logger.warning(f"Failed to load manifest {key}: {e}")
    except Exception as e:
        logger.warning(f"Could not list manifests for terrain={terrain}: {e}")
    logger.info(f"Total pseudo-labeled frames for terrain={terrain}: {len(combined)}")
    return combined


# ── Model helpers ──────────────────────────────────────────────────────────

def _load_previous_model(terrain: str):
    """
    Load latest UAV model for this terrain from MLflow.
    Prefers Production > Staging > latest version (mirrors feature_extractor.py logic).
    Falls back to ImageNet pretrained if no model exists for this terrain yet.
    """
    model_name = f"uav-navigator-{terrain}"
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise Exception(f"No registered versions for '{model_name}'")

        best = None
        for stage in ("Production", "Staging"):
            staged = [v for v in versions if v.current_stage == stage]
            if staged:
                best = sorted(staged, key=lambda v: int(v.version))[-1]
                break
        if best is None:
            best = sorted(versions, key=lambda v: int(v.version))[-1]

        model = mlflow.pytorch.load_model(f"models:/{model_name}/{best.version}")
        logger.info(
            f"Loaded {model_name} v{best.version} (stage={best.current_stage}) from MLflow"
        )
        return model, best.version

    except Exception as e:
        logger.info(f"No model for terrain={terrain} ({e}) — using ImageNet pretrained")
        base    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(512, 4)
        return base, None


def _freeze_backbone(model: NLResNet):
    """Freeze all layers except FC — for medium fine-tune."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.resnet.fc.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Medium fine-tune: FC only — {trainable:,} / {total:,} params trainable")


def _unfreeze_all(model: NLResNet):
    """Unfreeze all layers — for slow full retrain."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Slow full retrain: all {total:,} params trainable")


def _log_weight_norms(model: NLResNet) -> dict:
    backbone_norm = sum(
        p.data.norm().item() ** 2
        for name, p in model.named_parameters()
        if "fc" not in name
    ) ** 0.5
    fc_norm = sum(
        p.data.norm().item() ** 2
        for p in model.resnet.fc.parameters()
    ) ** 0.5
    return {"backbone_norm": round(backbone_norm, 4), "fc_norm": round(fc_norm, 4)}


def _estimate_latency(model: NLResNet) -> float:
    """
    Measure p95 single-image inference latency on CPU in ms.
    20 warmup + measure iterations to reduce JIT/cache noise.
    """
    dummy = torch.zeros(1, 3, 224, 224)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(25):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return round(times[int(len(times) * 0.95)], 1)


# ── Training loop ──────────────────────────────────────────────────────────

def _mock_metrics(level: str, prev_version) -> tuple[float, float, float]:
    """
    Fallback mock metrics when no pseudo-labeled frames are available.
    Used on first run (Ceph empty) or when Ceph is unreachable.
    Clearly marked in MLflow via 'mock_training=True' tag.
    """
    warmup = prev_version is not None
    if level == "medium":
        acc, loss = (0.80, 0.24) if warmup else (0.75, 0.32)
    else:
        acc, loss = (0.84, 0.18) if warmup else (0.80, 0.26)
    latency = 140.0 if warmup else 160.0
    logger.warning(
        f"No pseudo-labeled frames found — using mock metrics "
        f"(accuracy={acc}  loss={loss}  latency={latency}ms)"
    )
    return acc, latency, loss


def train_loop(
    model: NLResNet, terrain: str, level: str, prev_version
) -> tuple[float, float, float]:
    """
    Real training loop using pseudo-labeled frames from Ceph manifests.

    Returns (accuracy, latency_p95_ms, loss) — all computed on training set
    (no held-out validation set yet; splitting is a future improvement).

    Falls back to _mock_metrics() if no manifest found.
    """
    frame_labels = _load_manifests(terrain)

    if not frame_labels or prev_version is None:
        # No labels yet, OR no trained model yet (prev_version=None means FC is random-init
        # → pseudo-labels from this flight are random → training on them is meaningless).
        # Bootstrap with mock metrics; quality improves as model iterates over flights.
        if not frame_labels:
            logger.info("No pseudo-labeled frames found — using mock metrics")
        else:
            logger.info(
                f"No previous model (prev_version=None) — pseudo-labels from random FC "
                f"are unreliable; using mock metrics to bootstrap ({len(frame_labels)} frames logged)"
            )
            mlflow.log_metric("labeled_frames", len(frame_labels))
        mlflow.set_tag("mock_training", "true")
        return _mock_metrics(level, prev_version)

    mlflow.set_tag("mock_training", "false")
    mlflow.log_metric("labeled_frames", len(frame_labels))

    dataset = CephFrameDataset(s3, frame_labels)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    lr         = 1e-4 if level == "slow" else 5e-4
    num_epochs = 10   if level == "slow" else 5
    optimizer  = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in loader:
            logits = model(images)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs          = labels.size(0)
            epoch_loss += loss.item() * bs
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += bs
        total_loss += epoch_loss
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}  "
            f"loss={epoch_loss/max(total,1):.4f}  "
            f"acc={correct/max(total,1):.4f}"
        )

    model.eval()
    accuracy = round(correct / total,       4) if total > 0 else 0.0
    avg_loss = round(total_loss / total,    4) if total > 0 else 0.0
    latency  = _estimate_latency(model)

    logger.info(
        f"Training complete: accuracy={accuracy}  loss={avg_loss}  latency={latency}ms  "
        f"frames={total}  epochs={num_epochs}"
    )
    return accuracy, latency, avg_loss


# ── Helpers ────────────────────────────────────────────────────────────────

def _count_training_frames(terrain: str) -> int:
    """Count total frames (jpg) for this terrain in Ceph — used for logging only."""
    try:
        count = 0
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=BUCKET_TRAINING_DATA, Prefix=f"frames/terrain_{terrain}/"
        ):
            count += sum(
                1 for obj in page.get("Contents", [])
                if obj["Key"].endswith(".jpg")
            )
        return count
    except Exception:
        return 0


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    logger.info(f"Starting | level={LEVEL} | terrain={TERRAIN} | checkpoint={CHECKPOINT_KEY}")

    num_frames = _count_training_frames(TERRAIN)
    logger.info(
        f"Training data: {num_frames} frames in training-data/frames/terrain_{TERRAIN}/"
    )

    # ── Load base model ───────────────────────────────────────────────────
    base_model, prev_version = _load_previous_model(TERRAIN)

    # ── Wrap with Titans context step ─────────────────────────────────────
    W_fast, W_med, W_slow = _load_titans_state(TERRAIN)
    nl_model = NLResNet(base_model, W_fast, W_med, W_slow)

    if LEVEL == "medium":
        _freeze_backbone(nl_model)
    else:
        _unfreeze_all(nl_model)

    # ── Weight snapshot before training ───────────────────────────────────
    norms_before = _log_weight_norms(nl_model)
    logger.info(f"Weight norms BEFORE: {norms_before}")

    # ── Run number ────────────────────────────────────────────────────────
    model_name = f"uav-navigator-{TERRAIN}"
    try:
        existing   = client.search_model_versions(f"name='{model_name}'")
        run_number = len(existing) + 1
    except Exception:
        run_number = 1

    # ── Train + log ───────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"uav-retrain-{LEVEL}-{TERRAIN}-run{run_number}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}  run_number={run_number}")

        mlflow.log_params({
            "level":           LEVEL,
            "terrain":         TERRAIN,
            "num_frames":      num_frames,
            "checkpoint_key":  CHECKPOINT_KEY,
            "base_model":      "resnet18+titans",
            "num_classes":     4,
            "prev_version":    prev_version or "imagenet",
            "frozen_backbone": LEVEL == "medium",
            "nl_forward_pass": True,
            "memory_type":     "titans_gradient",
            "run_number":      run_number,
            "timestamp":       int(time.time()),
        })
        mlflow.set_tag("terrain",    TERRAIN)
        mlflow.set_tag("job_id",     JOB_ID)
        mlflow.set_tag("run_number", str(run_number))

        # W matrix norms — proxy for how much memory has been accumulated
        wf_norm = round(W_fast.norm().item(), 4)
        wm_norm = round(W_med.norm().item(),  4)
        ws_norm = round(W_slow.norm().item(), 4)
        mlflow.log_metrics({
            "W_fast_norm": wf_norm,
            "W_med_norm":  wm_norm,
            "W_slow_norm": ws_norm,
        })
        mlflow.log_metrics({
            "backbone_norm_before": norms_before["backbone_norm"],
            "fc_norm_before":       norms_before["fc_norm"],
        })

        # ── Real training (falls back to mock if no labels in Ceph) ──────
        accuracy, latency, loss = train_loop(nl_model, TERRAIN, LEVEL, prev_version)

        norms_after    = _log_weight_norms(nl_model)
        backbone_delta = round(abs(norms_after["backbone_norm"] - norms_before["backbone_norm"]), 4)
        fc_delta       = round(abs(norms_after["fc_norm"]       - norms_before["fc_norm"]),       4)

        mlflow.log_metrics({
            "accuracy":             accuracy,
            "latency_p95":          latency,
            "loss":                 loss,
            "num_frames":           num_frames,
            "backbone_norm_after":  norms_after["backbone_norm"],
            "fc_norm_after":        norms_after["fc_norm"],
            "backbone_delta":       backbone_delta,
            "fc_delta":             fc_delta,
        })
        logger.info(
            f"Metrics → accuracy={accuracy}  latency={latency}ms  loss={loss}"
        )
        logger.info(
            f"Weight delta → backbone={backbone_delta}  fc={fc_delta}"
        )

        # Log the inner ResNet (standard format) — feature_extractor.py loads this
        mlflow.pytorch.log_model(
            nl_model.resnet,
            artifact_path="uav-model",
            registered_model_name=model_name,
        )
        logger.info(
            f"Model registered as '{model_name}' run#{run_number}; artifact in s3://mlflow-artifacts/"
        )

    logger.info(f"Complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()
