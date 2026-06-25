"""
K8s Training Job — runs inside a K8s Pod when slow/medium level triggers.

Receives config from environment variables (injected via K8s Secret).
Logs metrics + registers model to MLflow; artifacts stored in Ceph S3.

NOTE: mock_train() returns simulated values for demo purposes.
      Replace with a real training loop once UAV ground-truth labels are available.
"""
import logging
import os
import random
import time

import boto3
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.models as models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [training_job] %(levelname)s %(message)s",
)
logger = logging.getLogger("training_job")

# ── Config from K8s env vars ───────────────────────────────────────────────
MLFLOW_URI     = os.environ.get("MLFLOW_TRACKING_URI",    "http://localhost:5000")
S3_ENDPOINT    = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:7480")
S3_ACCESS_KEY  = os.environ.get("AWS_ACCESS_KEY_ID",      "uavaccess")
S3_SECRET_KEY  = os.environ.get("AWS_SECRET_ACCESS_KEY",  "uavsecret123")
CHECKPOINT_KEY = os.environ.get("CHECKPOINT_KEY",         "slow/v1.pt")
LEVEL          = os.environ.get("TRAINING_LEVEL",         "slow")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]      = S3_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]  = S3_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"]     = "us-east-1"

mlflow.set_tracking_uri(MLFLOW_URI)

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name="us-east-1",
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _count_training_frames() -> int:
    try:
        resp = s3.list_objects_v2(Bucket="training-data", Prefix="frames/")
        return len(resp.get("Contents", []))
    except Exception:
        return 0


def mock_train(num_frames: int, level: str):
    """
    Mock training loop — simulates metrics for demo pipeline.
    Replace with a real training loop (dataloader + optimizer + loss) when
    UAV data with ground-truth labels is available.

    slow level trains longer → slightly higher accuracy than medium.
    """
    logger.info(f"Training on {num_frames} frames (level={level})...")
    time.sleep(3)  # simulate training time

    base_acc = 0.78 if level == "medium" else 0.82
    accuracy = round(base_acc + random.uniform(0.01, 0.08), 4)
    latency  = round(random.uniform(110, 185), 1)
    loss     = round(random.uniform(0.15, 0.35), 4)
    return accuracy, latency, loss


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    logger.info(f"Starting | level={LEVEL} | checkpoint={CHECKPOINT_KEY}")

    num_frames = _count_training_frames()
    logger.info(f"Training data: {num_frames} frames in training-data/")

    # ResNet18 with 4 output classes: forward / left / right / stop
    model    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 4)

    with mlflow.start_run(run_name=f"uav-retrain-{LEVEL}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        mlflow.log_params({
            "level":          LEVEL,
            "num_frames":     num_frames,
            "checkpoint_key": CHECKPOINT_KEY,
            "base_model":     "resnet18",
            "num_classes":    4,
            "timestamp":      int(time.time()),
        })

        accuracy, latency, loss = mock_train(num_frames, LEVEL)

        mlflow.log_metrics({
            "accuracy":    accuracy,
            "latency_p95": latency,
            "loss":        loss,
            "num_frames":  num_frames,
        })
        logger.info(f"Metrics → accuracy={accuracy}  latency={latency}ms  loss={loss}")

        mlflow.pytorch.log_model(
            model,
            artifact_path="uav-model",
            registered_model_name="uav-navigator",
        )
        logger.info("Model registered as 'uav-navigator'; artifact in s3://mlflow-artifacts/")

    logger.info(f"Complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()
