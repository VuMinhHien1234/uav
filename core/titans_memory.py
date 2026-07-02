
import io
import logging
import time

import torch
import torch.nn.functional as F

from config import settings
from infra.s3_client import make_s3_client

logger = logging.getLogger(__name__)


def _safe_norm(v: torch.Tensor, dim: int = 0) -> torch.Tensor:
    n = v.norm(dim=dim, keepdim=True).clamp(min=1e-8)
    return v / n


class _TitansScale:
    """
    Single-timescale Titans memory: W ∈ R^{d×d}.
    At inference time (inner loop):
      1. Normalize input x to unit sphere (epsilon-safe).
      2. Recall r = _safe_norm(W @ x).
      3. Compute surprise = MSE(r, x) before update — this is the validation signal.
      4. Gradient step with clipping: W ← forget·W − lr·clip(∇_W)
    """
    def __init__(self, embed_dim: int, lr: float, forget: float):
        self.W      = torch.zeros(embed_dim, embed_dim)
        self.lr     = lr
        self.forget = forget

    def recall_and_update(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Recall from memory + in-place gradient update.

        Args:
            x: feature vector (embed_dim,) — raw, normalized internally.
        Returns:
            (recall_vector, surprise_score)
            recall_vector : blended memory recall, same dim as x
            surprise_score: MSE before update in [0, 4/512] ≈ [0, 0.0078]
                            (0=familiar, ≈0.008=completely new/antiparallel)
        """
        x = _safe_norm(x.float().squeeze(), dim=0)   
        r = _safe_norm(self.W @ x, dim=0)            
        surprise = F.mse_loss(r, x).item()          
        residual = r - x
        grad_W   = torch.outer(residual, x)
        grad_norm = grad_W.norm().item()
        if grad_norm > 1.0:
            grad_W = grad_W / grad_norm  
        self.W = self.forget * self.W - self.lr * grad_W

        return r, surprise

    def recall(self, x: torch.Tensor) -> torch.Tensor:
        """Read-only recall (no update) — used during outer-loop training."""
        x = _safe_norm(x.float().squeeze(), dim=0)
        return _safe_norm(self.W @ x, dim=0)

    @property
    def norm(self) -> float:
        return round(self.W.norm().item(), 4)


class TitansMemory:
    """
    3-timescale Titans neural memory — drop-in replacement for VisualPathMemory.

    Public API (used by NestedLearningScheduler):
      recall_all(x)             → (combined_recall, surprise)  — recall + update all 3
      save_to_ceph()            — persist W matrices, cleanup old files
      cleanup_old_states(days)  — delete .pt files older than N days
    """

    def __init__(self, terrain: str = None, embed_dim: int = None, flight_id: int = None):
        self.s3        = make_s3_client()
        self.terrain   = terrain or settings.FLIGHT_TERRAIN
        self.flight_id = flight_id or int(time.time())
        dim            = embed_dim or settings.NL_EMBED_DIM

        self.fast = _TitansScale(dim, lr=settings.TITANS_LR_FAST,  forget=settings.NL_ALPHA_FAST)
        self.med  = _TitansScale(dim, lr=settings.TITANS_LR_MED,   forget=settings.NL_ALPHA_MEDIUM)
        self.slow = _TitansScale(dim, lr=settings.TITANS_LR_SLOW,  forget=settings.NL_ALPHA_SLOW_W)

        self._load_from_ceph()


    def recall_all(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Recall from all 3 timescales and update each in-place.

        Returns:
          combined_recall : blended recall (same dim as x)
          surprise        : weighted-average MSE across timescales
                            0.0   = completely familiar terrain
                            ~0.008 = new / significantly changed terrain (max for unit vectors in R^512)
        """
        x = x.float().squeeze()

        r_fast, s_fast = self.fast.recall_and_update(x)
        r_med,  s_med  = self.med.recall_and_update(x)
        r_slow, s_slow = self.slow.recall_and_update(x)

        combined = (
            settings.NL_BLEND_FAST   * r_fast
            + settings.NL_BLEND_MEDIUM * r_med
            + settings.NL_BLEND_SLOW_W * r_slow
        )

        total_blend = settings.NL_BLEND_FAST + settings.NL_BLEND_MEDIUM + settings.NL_BLEND_SLOW_W
        surprise = (
            settings.NL_BLEND_FAST   * s_fast
            + settings.NL_BLEND_MEDIUM * s_med
            + settings.NL_BLEND_SLOW_W * s_slow
        ) / total_blend

        return combined, surprise

    # ── Ceph persistence (same bucket + key structure as VisualPathMemory) ────

    def _ceph_prefix(self) -> str:
        return f"terrain_{self.terrain}"

    def save_to_ceph(self):
        """
        Persist this flight's W matrices to Ceph under a key unique to this
        flight: terrain_{t}/flight_{flight_id}.pt.

        Previously this wrote directly to terrain_{t}/latest.pt (last-write-
        wins): if two UAVs flying the same terrain saved concurrently, one
        flight's accumulated learning was silently overwritten by the other's
        — a race condition called out as an accepted limitation in earlier
        versions of this file. Writing to a per-flight key removes the race
        at the write level entirely — no two flights ever touch the same
        object, so there's nothing to overwrite.

        A separate aggregation step (core/titans_aggregate.py, run
        periodically by consumers/model_trainer_watcher.py) merges every
        flight's per-flight state for a terrain into a consolidated
        terrain_{t}/latest.pt, which is what _load_from_ceph() reads at the
        start of the next flight.
        """
        state = {
            "W_fast": self.fast.W,
            "W_med":  self.med.W,
            "W_slow": self.slow.W,
        }
        buf  = io.BytesIO()
        torch.save(state, buf)
        data = buf.getvalue()
        pfx  = self._ceph_prefix()
        key  = f"{pfx}/flight_{self.flight_id}.pt"

        try:
            self.s3.put_object(
                Bucket=settings.BUCKET_FAST_WEIGHT,
                Key=key,
                Body=data,
                ContentType="application/octet-stream",
            )
            logger.info(
                f"Titans saved → {key}  terrain={self.terrain}  "
                f"W_fast={self.fast.norm}  W_med={self.med.norm}  W_slow={self.slow.norm}"
            )
            # Cleanup old per-flight snapshots after every save (this flight's own)
            self.cleanup_old_states(keep_days=7)
        except Exception as e:
            logger.warning(f"Failed to save Titans memory to Ceph: {e}")

    def cleanup_old_states(self, keep_days: int = 7):
        """
        Delete per-flight .pt files (terrain_{t}/flight_{id}.pt) older than
        keep_days from Ceph. Skips latest.pt (the aggregator's consolidated
        output, always kept). Prevents unbounded growth of the
        fast-weight-state bucket as more flights accumulate.
        """
        cutoff = int(time.time()) - keep_days * 86400
        pfx    = self._ceph_prefix()
        try:
            resp = self.s3.list_objects_v2(
                Bucket=settings.BUCKET_FAST_WEIGHT, Prefix=pfx + "/"
            )
            deleted = 0
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith("latest.pt"):
                    continue  # always keep — this is the aggregator's output
                fname = key.split("/")[-1]
                if fname.startswith("flight_") and fname.endswith(".pt"):
                    fname = fname[len("flight_"):-len(".pt")]
                else:
                    fname = fname.replace(".pt", "")
                try:
                    ts = int(fname)
                    if ts < cutoff:
                        self.s3.delete_object(
                            Bucket=settings.BUCKET_FAST_WEIGHT, Key=key
                        )
                        deleted += 1
                except ValueError:
                    pass  # non-timestamp filename, skip
            if deleted:
                logger.info(f"Cleaned up {deleted} old Titans snapshots for terrain={self.terrain}")
        except Exception as e:
            logger.debug(f"Cleanup skipped: {e}")

    def _load_from_ceph(self):
        # latest.pt is written only by the aggregator (core/titans_aggregate.py),
        # never directly by a flight — see save_to_ceph() above.
        key = f"{self._ceph_prefix()}/latest.pt"
        try:
            obj    = self.s3.get_object(Bucket=settings.BUCKET_FAST_WEIGHT, Key=key)
            buf    = io.BytesIO(obj["Body"].read())
            loaded = torch.load(buf, weights_only=True)

            if "W_fast" in loaded:
                # Native Titans format
                self.fast.W = loaded["W_fast"]
                self.med.W  = loaded["W_med"]
                self.slow.W = loaded["W_slow"]
                logger.info(
                    f"Titans restored  terrain={self.terrain}  "
                    f"W_fast={self.fast.norm}  W_med={self.med.norm}  W_slow={self.slow.norm}"
                )
        except Exception:
            logger.info(
                f"No Titans state in Ceph for terrain={self.terrain} — starting fresh (zeros)"
            )
