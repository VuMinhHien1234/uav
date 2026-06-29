"""
NestedLearningScheduler — Titans edition.

Replaces blind Hebbian memory update with gradient-based Titans memory.

Key changes vs Hebbian edition:
  - Memory update happens INSIDE recall_all() — no separate update_memory() call needed.
  - delta is now Titans' surprise score (MSE reconstruction error) instead of cosine distance.
  - update_memory() removed — flight_agent no longer needs action_emb for memory update.

Surprise score semantics (same as old delta semantics):
  low surprise  → terrain is familiar → SKIP / FAST
  high surprise → terrain is new      → MEDIUM / SLOW → trigger Kafka retrain event
"""

import logging

import torch
import torch.nn.functional as F

from config import settings
from core.titans_memory import TitansMemory

logger = logging.getLogger(__name__)


class NestedLearningScheduler:

    def __init__(self, terrain: str = None):
        self.memory            = TitansMemory(terrain=terrain)
        self.drift_accumulator = 0.0
        self.last_combined: torch.Tensor | None = None

    def decide(self, current_features: torch.Tensor, frame_id: str = None):
        """
        Xử lý một frame:
          1. Truy hồi từ Titans memory + cập nhật gradient tại chỗ (inner loop).
          2. Xây dựng vector tổng hợp: input + recall.
          3. Phân loại mức độ theo surprise score (thay thế cosine delta).
          4. Tích lũy drift để kích hoạt slow trigger.

        Trả về (level, debug_dict) — API giữ nguyên so với phiên bản Hebbian.
        """
        x = current_features.float().squeeze()

        # Inner loop: recall + gradient update, returns (combined_recall, surprise)
        recall, surprise = self.memory.recall_all(x)

        # Combined: input enriched by Titans memory
        combined = F.normalize(x + recall, dim=0)
        self.last_combined = combined.detach()

        # Surprise replaces cosine delta — same threshold semantics
        delta = surprise
        self.drift_accumulator += delta

        debug = {
            "level":     "?",
            "delta":     round(delta,                    4),
            "surprise":  round(surprise,                 4),
            "drift_acc": round(self.drift_accumulator,   2),
            "wf_norm":   round(self.memory.fast.norm,    4),
            "wm_norm":   round(self.memory.med.norm,     4),
            "ws_norm":   round(self.memory.slow.norm,    4),
        }

        if self.drift_accumulator >= settings.NL_SLOW_ACCUMULATOR:
            self.drift_accumulator = 0.0
            return "slow",   {**debug, "level": "slow"}
        if delta >= settings.NL_MEDIUM_DELTA:
            return "medium", {**debug, "level": "medium"}
        if delta >= settings.NL_FAST_DELTA:
            return "fast",   {**debug, "level": "fast"}
        return "skip",   {**debug, "level": "skip"}
