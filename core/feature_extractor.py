"""
ResNet18 feature extractor — shared between UC1 and UC3.
Extracted once here so both consumers load the same code path.
"""
import io
import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FeatureExtractor:
    """Wraps a pretrained ResNet18 (FC removed) for 512-dim embedding extraction."""

    def __init__(self):
        logger.info("Loading ResNet18 feature extractor...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()
        logger.info("ResNet18 ready.")

    def from_bytes(self, img_bytes: bytes) -> list:
        """Extract embedding from raw image bytes. Returns list[float] of 512 dims."""
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self._run(_TRANSFORM(img).unsqueeze(0)).tolist()

    def from_path(self, img_path: str) -> torch.Tensor:
        """Extract embedding from image file. Returns tensor of shape (512,)."""
        img = Image.open(img_path).convert("RGB")
        return self._run(_TRANSFORM(img).unsqueeze(0))

    def _run(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(tensor).squeeze()
