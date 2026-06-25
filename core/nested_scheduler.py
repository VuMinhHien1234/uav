"""
NestedLearningScheduler — per-frame decision: fast / medium / slow / memory_hit / skip.

Decision logic (NeurIPS 2025 Nested Learning pattern):
  memory_hit — cosine_sim with stored frame > NL_MEMORY_THRESHOLD → recall action
  slow       — accumulated drift >= NL_SLOW_ACCUMULATOR → full retrain
  medium     — single-frame delta >= NL_MEDIUM_DELTA    → fine-tune
  fast       — single-frame delta >= NL_FAST_DELTA      → local adapt + save memory
  skip       — delta below all thresholds               → no action
"""


import logging

import torch
import torch.nn.functional as F

from config import settings
from core.path_memory import VisualPathMemory

logger = logging.getLogger(__name__)


class NestedLearningScheduler:
    
    def __init__(self):
        self.memory            = VisualPathMemory()
        self.drift_accumulator = 0.0
        self.prev_features: torch.Tensor | None = None

    def decide(self, current_features: torch.Tensor, frame_id: str = None):
        """
        Args:
            current_features: tensor of shape (512,) or (1, 512)
            frame_id:         optional identifier for logging

        Returns:
            (level, recalled_action_or_None, debug_info)
        """
        cur = current_features.float().squeeze()

        # ── Step 0: check path memory ──────────────────────────────────────
        recalled_action, confidence = self.memory.recall(cur.tolist())
        if recalled_action is not None:
            return "memory_hit", recalled_action, {
                "level":      "memory_hit",
                "confidence": round(confidence, 3),
                "drift_acc":  round(self.drift_accumulator, 2),
            }

        # ── Step 1: compute delta vs previous frame ────────────────────────
        if self.prev_features is None:
            self.prev_features = cur.detach()
            return "fast", None, {"level": "fast", "delta": 0.0, "drift_acc": 0.0}

        sim   = F.cosine_similarity(cur.unsqueeze(0), self.prev_features.unsqueeze(0)).item()
        delta = 1.0 - sim
        self.drift_accumulator += delta
        self.prev_features = cur.detach()

        debug = {
            "delta":    round(delta, 3),
            "drift_acc": round(self.drift_accumulator, 2),
            "sim":      round(sim, 3),
        }

        # ── Step 2: decide level ───────────────────────────────────────────
        if self.drift_accumulator >= settings.NL_SLOW_ACCUMULATOR:
            self.drift_accumulator = 0.0
            return "slow", None, {**debug, "level": "slow"}
        if delta >= settings.NL_MEDIUM_DELTA:
            return "medium", None, {**debug, "level": "medium"}
        if delta >= settings.NL_FAST_DELTA:
            return "fast", None, {**debug, "level": "fast"}
        return "skip", None, {**debug, "level": "skip"}

    def save_memory(
        self,
        frame_id: str,
        features: torch.Tensor,
        action: str = "straight",
        reward: float = 0.5,
    ):
        self.memory.remember(frame_id, features.squeeze(), action, reward)
