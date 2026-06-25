
"""
VisualPathMemory — store and recall visual embeddings from Ceph S3.
UAV "nhớ đường" bằng cách so sánh cosine similarity giữa các frame.
"""


import json
import logging
import time

import torch
import torch.nn.functional as F

from config import settings
from infra.s3_client import make_s3_client

logger = logging.getLogger(__name__)


class VisualPathMemory:
    def __init__(self, max_cache: int = 200):
        self.s3        = make_s3_client()
        self.cache: dict = {}
        self.max_cache = max_cache
        self._load_from_ceph()

    def _load_from_ceph(self):
        try:
            resp  = self.s3.list_objects_v2(
                Bucket=settings.BUCKET_EMBEDDINGS, Prefix="path-memory/"
            )
            items = resp.get("Contents", [])[: self.max_cache]
            for item in items:
                obj = self.s3.get_object(
                    Bucket=settings.BUCKET_EMBEDDINGS, Key=item["Key"]
                )
                mem = json.loads(obj["Body"].read())
                self.cache[mem["frame_id"]] = mem
            logger.info(f"Loaded {len(self.cache)} entries from Ceph path-memory")
        except Exception as e:
            logger.warning(f"Path memory empty (first run): {e}")

    def remember(
        self,
        frame_id: str,
        embedding,
        action: str = "straight",
        reward: float = 0.5,
    ):
        """Persist embedding to Ceph and local cache."""
        entry = {
            "frame_id":  frame_id,
            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            "action":    action,
            "reward":    reward,
            "timestamp": time.time(),
        }
        self.s3.put_object(
            Bucket=settings.BUCKET_EMBEDDINGS,
            Key=f"path-memory/{frame_id}.json",
            Body=json.dumps(entry),
        )
        self.cache[frame_id] = entry

    def recall(self, current_emb, threshold: float = None):
        """
        Find nearest neighbour by cosine similarity.
        Returns (action, confidence) or (None, 0.0) if no match above threshold.
        """
        threshold = threshold if threshold is not None else settings.NL_MEMORY_THRESHOLD
        if not self.cache:
            return None, 0.0

        cur      = torch.tensor(current_emb).float().unsqueeze(0)
        best_sim = 0.0
        best_action = None

        for mem in self.cache.values():
            stored = torch.tensor(mem["embedding"]).float().unsqueeze(0)
            sim    = F.cosine_similarity(cur, stored).item()
            if sim > best_sim:
                best_sim    = sim
                best_action = mem["action"]

        if best_sim >= threshold:
            return best_action, best_sim
        return None, best_sim

    def count(self) -> int:
        return len(self.cache)
