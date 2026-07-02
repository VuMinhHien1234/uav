"""
model_trainer.py — Kafka consumer for retrain_trigger events.

Fast, non-blocking: reads an event, dedup-checks it, spawns a K8s training
Job, hands the job off to the watcher (model_trainer_watcher.py), and reads
the next message immediately. It does NOT wait for the Job to finish —
that used to happen inline here (via wait_for_k8s_job / wait_for_mlflow_run),
which meant one event could occupy this loop for up to ~15 minutes before
the next event was even read. See _trainer_common.py for why the split
works the way it does and how the two processes hand off through Ceph
markers instead of shared memory.

Run both processes together:
    python3 -m consumers.model_trainer            # this file — ingestion
    python3 -m consumers.model_trainer_watcher     # completion + promotion

To scale ingestion throughput: increase partitions on the uav-retrain topic
and run more instances of this file with the same group_id — Kafka will
split partitions across them automatically. (Requires >1 partition; see
project notes on the current single-partition Kafka setup.)
"""
import json
import logging
import time

from config import settings
from infra.kafka_consumer import make_consumer
from infra.s3_client import make_s3_client

from consumers._trainer_common import (
    is_already_processed,
    mark_pending,
    spawn_k8s_training_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [model_trainer] %(levelname)s %(message)s",
)
logger = logging.getLogger("model_trainer")


def main():
    s3       = make_s3_client()
    consumer = make_consumer(topic=settings.KAFKA_TOPIC_RETRAIN, group_id="model-trainer-group")
    logger.info(f"Ready — listening on topic={settings.KAFKA_TOPIC_RETRAIN}")

    for msg in consumer:
        event = msg.value

        if event.get("event") != "retrain_trigger":
            logger.debug(f"Skipping non-retrain event: {event.get('event')}")
            continue

        key   = event.get("checkpoint_key", "")
        level = event.get("level", "slow")

        # ── Dedup via Ceph marker (shared across instances + restarts) ─────
        if is_already_processed(s3, key):
            logger.info(f"Already processed key={key} — skipping (Ceph marker found)")
            continue

        logger.info(f"Retrain event received: level={level}  key={key}")

        # ── Read training metadata from Ceph ──────────────────────────────
        try:
            obj             = s3.get_object(Bucket=settings.BUCKET_CHECKPOINTS, Key=key)
            checkpoint_info = json.loads(obj["Body"].read())
        except Exception:
            logger.exception(f"Cannot read checkpoint {key}")
            continue

        terrain = checkpoint_info.get("terrain")
        if not terrain:
            logger.warning(
                f"Checkpoint {key} missing 'terrain' field — skipping to avoid "
                f"training on wrong terrain data"
            )
            continue
        logger.info(f"Terrain: {terrain}")

        job_id = f"uav-train-{int(time.time())}"
        try:
            spawn_k8s_training_job(job_id, level, key, terrain)
        except Exception:
            logger.exception("Failed to spawn K8s Job")
            continue

        # Hand off to the watcher and move on immediately — no blocking wait
        # for the Job or the MLflow run here.
        mark_pending(s3, job_id, key, terrain, level)
        logger.info(f"Job {job_id} spawned for key={key} — handed off to watcher")


if __name__ == "__main__":
    main()
