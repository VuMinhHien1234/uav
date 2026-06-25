"""
UC3 Simulator — UAV Frame Generator + NestedLearningScheduler

Simulates UAV sending frames to Ceph. The scheduler decides the learning level;
slow/medium uploads a checkpoint to trigger UC2 → K8s retraining.

In production: runs on Jetson Nano onboard the UAV.
For demo:      reads frames from mock-data/ on a VM.
"""
import glob
import json
import logging
import os
import random
import time

from config import settings
from core.feature_extractor import FeatureExtractor
from core.nested_scheduler import NestedLearningScheduler
from infra.s3_client import make_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [uc3] %(levelname)s %(message)s",
)
logger = logging.getLogger("uc3")

LEVEL_ICON = {
    "memory_hit": "🔵",
    "slow":       "🔴",
    "medium":     "🟡",
    "fast":       "🟢",
    "skip":       "⚫",
}


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _upload_frame(s3, img_path: str, frame_id: str) -> str:
    key = f"env/{frame_id}.jpg"
    with open(img_path, "rb") as f:
        s3.put_object(
            Bucket=settings.BUCKET_RAW_FRAMES,
            Key=key,
            Body=f.read(),
            ContentType="image/jpeg",
        )
    return key


def _upload_checkpoint(s3, level: str, frame_id: str, debug_info: dict) -> str:
    checkpoint = {
        "level":      level,
        "frame_id":   frame_id,
        "drift_info": debug_info,
        "timestamp":  time.time(),
        "action":     "trigger_retrain",
    }
    key = f"{level}/uav_{level}_{int(time.time())}.json"
    s3.put_object(
        Bucket=settings.BUCKET_CHECKPOINTS,
        Key=key,
        Body=json.dumps(checkpoint),
        ContentType="application/json",
    )
    return key


# ── Frame source ──────────────────────────────────────────────────────────────

def _get_mock_frames() -> list:
    frames = []
    for env in ["env1", "env2", "env3"]:
        frames.extend(sorted(glob.glob(f"mock-data/{env}/*.jpg")))
    if not frames:
        raise FileNotFoundError(
            "No frames in mock-data/. Run: python3 scripts/download_mock_data.py"
        )
    return frames


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    s3        = make_s3_client()
    extractor = FeatureExtractor()
    scheduler = NestedLearningScheduler()

    frame_paths      = _get_mock_frames()
    checkpoint_count = 0
    logger.info(f"Found {len(frame_paths)} frames across 3 environments — starting simulation")
    print("=" * 60)

    for i, img_path in enumerate(frame_paths):
        frame_id = f"frame_{i:04d}_{os.path.basename(img_path).replace('.jpg', '')}"
        env      = img_path.split("/")[1]

        features = extractor.from_path(img_path)
        level, recalled_action, debug = scheduler.decide(features, frame_id)

        icon = LEVEL_ICON.get(level, "?")
        logger.info(
            f"Frame {i:04d} | {env} | {icon} {level.upper():12s} | "
            f"delta={debug.get('delta', 0):.3f}  drift={debug.get('drift_acc', 0):.1f}"
        )

        if level == "skip":
            time.sleep(settings.FRAME_DELAY)
            continue

        if level == "memory_hit":
            logger.info(
                f"Recalled action: '{recalled_action}' (conf={debug['confidence']:.3f})"
            )
            time.sleep(settings.FRAME_DELAY)
            continue

        _upload_frame(s3, img_path, frame_id)
        logger.debug(f"Uploaded → raw-frames/env/{frame_id}.jpg")

        if level == "fast":
            action = random.choice(["straight", "left", "right"])
            reward = round(random.uniform(0.3, 0.8), 2)
            scheduler.save_memory(frame_id, features, action, reward)
            logger.info(f"Memory saved: action={action}  reward={reward}")

        elif level in ("slow", "medium"):
            checkpoint_count += 1
            key = _upload_checkpoint(s3, level, frame_id, debug)
            logger.info(f"Checkpoint #{checkpoint_count} → checkpoints/{key}")

        time.sleep(settings.FRAME_DELAY)

    print("=" * 60)
    logger.info(
        f"Simulation complete: {len(frame_paths)} frames, "
        f"{checkpoint_count} checkpoints, "
        f"{scheduler.memory.count()} memory entries"
    )


if __name__ == "__main__":
    main()
