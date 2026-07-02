"""
core/titans_aggregate.py — merges per-flight Titans memory states into a
single consolidated terrain_{t}/latest.pt.

Background: core/titans_memory.py used to have every flight overwrite
terrain_{t}/latest.pt directly. Two UAVs flying the same terrain at once
would race on that single object — whichever flight's save_to_ceph() ran
last silently discarded the other's accumulated learning. Flights now save
to their own key instead (terrain_{t}/flight_{flight_id}.pt), so there is
no write-write race anymore. This module does the merging step: read every
flight's state for a terrain, average them, write the result out as the
new latest.pt that the next flight of that terrain will load.

Run periodically — consumers/model_trainer_watcher.py calls
aggregate_all_terrains() once every few poll cycles.
"""
import io
import logging

import torch

from config import settings

logger = logging.getLogger(__name__)


def _list_terrains(s3) -> list[str]:
    """Return distinct terrain names that have any state under fast-weight-state/."""
    terrains = set()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.BUCKET_FAST_WEIGHT, Delimiter="/"):
            for prefix in page.get("CommonPrefixes", []):
                name = prefix["Prefix"].rstrip("/")  # e.g. "terrain_forest"
                if name.startswith("terrain_"):
                    terrains.add(name[len("terrain_"):])
    except Exception:
        logger.exception("Could not list terrains under fast-weight-state/")
    return sorted(terrains)


def _list_flight_states(s3, terrain: str) -> list[str]:
    """Return keys of every per-flight state file for a terrain."""
    keys = []
    prefix = f"terrain_{terrain}/flight_"
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.BUCKET_FAST_WEIGHT, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
    except Exception:
        logger.exception(f"Could not list flight states for terrain={terrain}")
    return keys


def aggregate_terrain(s3, terrain: str, embed_dim: int = None) -> bool:
    """
    Merge every per-flight Titans state for one terrain into a new latest.pt.

    Merge strategy: element-wise mean across all contributing flights' W
    matrices. This is a pragmatic middle ground — not true sample-weighted
    federated averaging — but every flight's contribution is reflected in
    the result instead of only the most-recently-saved flight winning.

    Returns True if a new latest.pt was written, False if there were no
    flight states to aggregate yet for this terrain.
    """
    dim  = embed_dim or settings.NL_EMBED_DIM
    keys = _list_flight_states(s3, terrain)
    if not keys:
        return False

    sums = {
        "W_fast": torch.zeros(dim, dim),
        "W_med":  torch.zeros(dim, dim),
        "W_slow": torch.zeros(dim, dim),
    }
    count = 0
    for key in keys:
        try:
            obj   = s3.get_object(Bucket=settings.BUCKET_FAST_WEIGHT, Key=key)
            buf   = io.BytesIO(obj["Body"].read())
            state = torch.load(buf, weights_only=True)
            for name in sums:
                if name in state:
                    sums[name] += state[name]
            count += 1
        except Exception:
            logger.warning(f"Could not load flight state {key} — skipping")

    if count == 0:
        return False

    merged = {name: (tensor / count) for name, tensor in sums.items()}

    buf = io.BytesIO()
    torch.save(merged, buf)
    try:
        s3.put_object(
            Bucket=settings.BUCKET_FAST_WEIGHT,
            Key=f"terrain_{terrain}/latest.pt",
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        logger.info(f"Aggregated {count} flight state(s) for terrain={terrain} -> latest.pt")
        return True
    except Exception:
        logger.exception(f"Could not write aggregated latest.pt for terrain={terrain}")
        return False


def aggregate_all_terrains(s3):
    """Aggregate every terrain that currently has per-flight state files."""
    for terrain in _list_terrains(s3):
        try:
            aggregate_terrain(s3, terrain)
        except Exception:
            logger.exception(f"Aggregation failed for terrain={terrain}")
