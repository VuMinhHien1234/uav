import logging
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _find_best_version(versions):
    for stage in ("Production", "Staging"):
        staged = [v for v in versions if v.current_stage == stage]
        if staged:
            best = sorted(staged, key=lambda v: int(v.version))[-1]
            logger.info(
                f"Selected model v{best.version} from stage={stage}"
            )
            return best
    # No staged version — fall back to highest version number
    best = sorted(versions, key=lambda v: int(v.version))[-1]
    logger.info(
        f"No Production/Staging version found — falling back to latest v{best.version}"
    )
    return best


def _load_model(terrain: str) -> tuple[nn.Module, nn.Linear]:
    os.environ.setdefault("MLFLOW_TRACKING_URI",    settings.MLFLOW_URI)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.MLFLOW_S3_ENDPOINT)
    os.environ.setdefault("AWS_ACCESS_KEY_ID",       settings.S3_ACCESS_KEY)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY",   settings.S3_SECRET_KEY)
    os.environ.setdefault("AWS_DEFAULT_REGION",      settings.S3_REGION)

    model_name = f"uav-navigator-{terrain}"
    try:
        import mlflow.pytorch
        from mlflow import MlflowClient

        client   = MlflowClient(tracking_uri=settings.MLFLOW_URI)
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            raise LookupError(f"No registered versions for '{model_name}'")

        best     = _find_best_version(versions)
        full     = mlflow.pytorch.load_model(f"models:/{model_name}/{best.version}")
        backbone = nn.Sequential(*list(full.children())[:-1])
        fc       = full.fc
        logger.info(
            f"Loaded {model_name} v{best.version} "
            f"(stage={best.current_stage}) from MLflow"
        )
        return backbone, fc

    except Exception as e:
        logger.info(
            f"No model for terrain={terrain} in MLflow ({e}) — "
            f"using ImageNet pretrained ResNet18 as fallback"
        )
        resnet    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Linear(512, 4)  # replace 1000-class ImageNet head with 4-class UAV head
        backbone  = nn.Sequential(*list(resnet.children())[:-1])
        fc        = resnet.fc
        return backbone, fc

_ACTIONS = ["straight", "left", "right", "stop"]

class FeatureExtractor:
    def __init__(self, terrain: str = None):
        self.terrain = terrain if terrain is not None else settings.FLIGHT_TERRAIN
        logger.info(f"Initialising FeatureExtractor  terrain={self.terrain}...")
        self.backbone, self.fc = _load_model(self.terrain)
        self.backbone.eval()
        self.fc.eval()
        logger.info("FeatureExtractor ready.")

    def from_path(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        return self._run(_TRANSFORM(img).unsqueeze(0))

    def predict_action(self, combined: torch.Tensor) -> str:
        with torch.no_grad():
            logits = self.fc(combined.squeeze())
            idx    = logits.argmax().item()
        return _ACTIONS[idx]

    def _run(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(tensor).squeeze()
