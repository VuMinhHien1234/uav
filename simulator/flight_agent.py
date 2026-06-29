
import glob
import json
import logging
import os
import time

from config import settings
from core.feature_extractor import FeatureExtractor
from core.nested_scheduler import NestedLearningScheduler
from infra.kafka_producer import make_producer, publish_retrain_event
from infra.s3_client import make_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [flight_agent] %(levelname)s %(message)s",
)
logger = logging.getLogger("flight_agent")

LEVEL_ICON = {
    "slow":   "🔴",
    "medium": "🟡",
    "fast":   "🟢",
    "skip":   "⚫",
}

# ── S3 helpers ────────────────────────────────────────────────────────────────

def _upload_frame(s3, img_path: str, frame_id: str, flight_id: int, terrain: str) -> str:
    key = f"frames/terrain_{terrain}/flight_{flight_id}/{frame_id}.jpg"
    with open(img_path, "rb") as f:
        s3.put_object(
            Bucket=settings.BUCKET_TRAINING_DATA,
            Key=key,
            Body=f.read(),
            ContentType="image/jpeg",
        )
    return key


def _upload_checkpoint(
    s3, level: str, frame_id: str, flight_id: int, terrain: str,
    debug_info: dict, nav_action: str,
) -> tuple[str, float]:
    ts = time.time()
    checkpoint = {
        "level":      level,
        "frame_id":   frame_id,
        "flight_id":  flight_id,
        "terrain":    terrain,
        "drift_info": debug_info,
        "timestamp":  ts,
        "nav_action": nav_action,   # predicted navigation action (pseudo-label)
        "action":     "trigger_retrain",  # event type — kept for backward compat
    }
    key = f"{level}/terrain_{terrain}/flight_{flight_id}_{int(ts)}.json"
    s3.put_object(
        Bucket=settings.BUCKET_CHECKPOINTS,
        Key=key,
        Body=json.dumps(checkpoint),
        ContentType="application/json",
    )
    return key, ts


def _upload_flight_manifest(
    s3, terrain: str, flight_id: int, frame_labels: dict[str, str]
):
    """
    Upload a manifest mapping Ceph frame keys → predicted action (pseudo-label).

    Schema: {"frames/terrain_{t}/flight_{id}/{frame_id}.jpg": "straight", ...}

    Used by training/job.py to build a labeled dataset for real training.
    Pseudo-labels come from the model's own predictions (self-training).
    Only slow/medium frames are included — fast/skip frames are not uploaded to Ceph.
    """
    if not frame_labels:
        return
    key = f"frames/terrain_{terrain}/flight_{flight_id}/manifest.json"
    s3.put_object(
        Bucket=settings.BUCKET_TRAINING_DATA,
        Key=key,
        Body=json.dumps(frame_labels),
        ContentType="application/json",
    )
    logger.info(
        f"Manifest uploaded → training-data/{key}  ({len(frame_labels)} pseudo-labeled frames)"
    )


# ── Frame source ──────────────────────────────────────────────────────────────

def _get_mock_frames(terrain: str) -> list:
    """Return frames for a single terrain: mock-data/{terrain}/*.jpg"""
    frames = sorted(glob.glob(f"mock-data/{terrain}/*.jpg"))
    if not frames:
        available = [
            os.path.basename(d.rstrip("/"))
            for d in sorted(glob.glob("mock-data/*/"))
        ]
        raise FileNotFoundError(
            f"No frames for terrain='{terrain}' in mock-data/.\n"
            f"Available: {available}\n"
            f"Run: python3 scripts/download_mock_data.py"
        )
    return frames


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Unique ID + terrain must be set BEFORE extractor/scheduler (both need terrain)
    flight_id      = int(time.time())
    flight_terrain = settings.FLIGHT_TERRAIN   # override with: FLIGHT_TERRAIN=forest python3 -m simulator.flight_agent

    s3        = make_s3_client()
    producer  = make_producer()
    extractor = FeatureExtractor(terrain=flight_terrain)
    scheduler = NestedLearningScheduler(terrain=flight_terrain)

    frame_paths      = _get_mock_frames(flight_terrain)
    checkpoint_count = 0
    frame_labels: dict[str, str] = {}   # frame S3 key → predicted action (pseudo-labels)

    logger.info(
        f"Flight {flight_id} | terrain={flight_terrain} | "
        f"{len(frame_paths)} frames — starting simulation"
    )
    print("=" * 60)

    for i, img_path in enumerate(frame_paths):
        frame_id = f"frame_{i:04d}_{os.path.basename(img_path).replace('.jpg', '')}"
        features = extractor.from_path(img_path)
        level, debug = scheduler.decide(features, frame_id)
        icon = LEVEL_ICON.get(level, "?")
        logger.info(
            f"Frame {i:04d} | {flight_terrain} | {icon} {level.upper():8s} | "
            f"surprise={debug.get('surprise', 0):.4f}  drift={debug.get('drift_acc', 0):.1f}  "
            f"Wf={debug.get('wf_norm', 0):.4f}  "
            f"Wm={debug.get('wm_norm', 0):.4f}  "
            f"Ws={debug.get('ws_norm', 0):.4f}"
        )

        # ── Periodic Titans checkpoint (every 50 frames) ──────────────────
        if i > 0 and i % 50 == 0:
            scheduler.memory.save_to_ceph()
            logger.info(f"Periodic Titans save at frame {i}")

        if level == "skip":
            time.sleep(settings.FRAME_DELAY)
            continue

        # ── NL inference: combined → FC → action (all non-skip levels) ───
        # Titans: memory update already happened inside scheduler.decide() — no
        # separate update_memory() call needed, action_emb not required.
        action = extractor.predict_action(scheduler.last_combined)
        logger.info(f"Action: {action}  level={level}  surprise={debug.get('surprise', 0):.4f}")

        # ── Upload slow/medium frames directly to training-data ───────────
        # Fast frames: Titans update only — no Ceph upload needed for training.
        # In production: terrain derived from GPS + map data mid-flight.
        if level in ("slow", "medium"):
            checkpoint_count += 1
            frame_key = _upload_frame(s3, img_path, frame_id, flight_id, flight_terrain)
            frame_labels[frame_key] = action        # pseudo-label for this frame
            logger.debug(f"Uploaded → training-data/{frame_key}  label={action}")
            key, ts = _upload_checkpoint(
                s3, level, frame_id, flight_id, flight_terrain, debug, action
            )
            logger.info(f"Checkpoint #{checkpoint_count} → checkpoints/{key}")
            publish_retrain_event(producer, level, key, frame_id, ts)
        time.sleep(settings.FRAME_DELAY)

    # ── End of flight: persist Titans state + upload pseudo-label manifest ─
    scheduler.memory.save_to_ceph()
    _upload_flight_manifest(s3, flight_terrain, flight_id, frame_labels)

    print("=" * 60)
    logger.info(
        f"Flight {flight_id} complete: {len(frame_paths)} frames, "
        f"{checkpoint_count} checkpoints uploaded to training-data/frames/terrain_*/flight_{flight_id}/, "
        f"W_fast={scheduler.memory.fast.norm}  "
        f"W_med={scheduler.memory.med.norm}  "
        f"W_slow={scheduler.memory.slow.norm}"
    )


if __name__ == "__main__":
    main()
