"""
Shared helpers between consumers/model_trainer.py (fast, non-blocking Kafka
consumer) and consumers/model_trainer_watcher.py (separate poll loop that
waits for K8s Jobs to finish and promotes the resulting MLflow model).

Why this module exists:
  The old model_trainer.py did everything in one blocking loop — read a
  Kafka message, spawn a K8s Job, then block for up to ~15 minutes waiting
  for that Job (and the MLflow run after it) before reading the next
  message. That meant one retrain event could stall every other event
  behind it, even with more Kafka partitions/consumer instances.

  The fix splits the work into two independent processes that only ever
  talk to each other through markers stored on Ceph (not shared memory,
  not a direct call) — so either process can restart without losing track
  of in-flight jobs, and can be scaled/restarted independently:

    consumer (model_trainer.py):
      read event -> dedup check (processed/<checkpoint_key>) -> spawn Job
      -> write pending/<job_id>.json -> read next message immediately

    watcher (model_trainer_watcher.py):
      poll pending/*.json every few seconds -> check Job status (single,
      non-blocking check, not a retry loop) -> once Complete, look up the
      MLflow run once, validate + promote -> write processed/<key> ->
      delete pending/<job_id>.json
"""
import json
import logging
import os
import subprocess
import tempfile
import time

import mlflow
from mlflow import MlflowClient

from config import settings

logger = logging.getLogger("model_trainer")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]      = settings.S3_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]  = settings.S3_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"]     = settings.S3_REGION

mlflow.set_tracking_uri(settings.MLFLOW_URI)
client = MlflowClient()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K8S_JOB_YAML = os.path.join(_PROJECT_ROOT, "k8s", "training_job.yaml")


# ── Dedup markers ────────────────────────────────────────────────────────────
# Replaces the old in-memory `_seen_keys: set()`, which lived in one process's
# RAM only — broke as soon as you ran more than one consumer instance (each
# had its own set, so the same checkpoint could be trained twice). Markers on
# Ceph are shared across every consumer/watcher instance and survive restarts.

def _processed_marker_key(checkpoint_key: str) -> str:
    return f"processed/{checkpoint_key}"


def is_already_processed(s3, checkpoint_key: str) -> bool:
    try:
        s3.head_object(
            Bucket=settings.BUCKET_CHECKPOINTS, Key=_processed_marker_key(checkpoint_key)
        )
        return True
    except Exception:
        return False


def mark_processed(s3, checkpoint_key: str, status: str = "done"):
    try:
        s3.put_object(
            Bucket=settings.BUCKET_CHECKPOINTS,
            Key=_processed_marker_key(checkpoint_key),
            Body=json.dumps({"status": status, "ts": time.time()}),
            ContentType="application/json",
        )
    except Exception:
        logger.exception(f"Could not write processed marker for {checkpoint_key}")


# ── Pending-job tracking ─────────────────────────────────────────────────────
# Lets the watcher discover in-flight jobs (including after its own restart)
# without any in-memory state shared with the consumer process.

def _pending_marker_key(job_id: str) -> str:
    return f"pending/{job_id}.json"


def mark_pending(s3, job_id: str, checkpoint_key: str, terrain: str, level: str):
    try:
        s3.put_object(
            Bucket=settings.BUCKET_CHECKPOINTS,
            Key=_pending_marker_key(job_id),
            Body=json.dumps({
                "job_id":         job_id,
                "checkpoint_key": checkpoint_key,
                "terrain":        terrain,
                "level":          level,
                "spawned_at":     time.time(),
            }),
            ContentType="application/json",
        )
    except Exception:
        logger.exception(f"Could not write pending marker for job {job_id}")


def list_pending_jobs(s3) -> list[dict]:
    """Return all pending-job records currently tracked in Ceph."""
    pending = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.BUCKET_CHECKPOINTS, Prefix="pending/"):
            for obj in page.get("Contents", []):
                try:
                    body = s3.get_object(Bucket=settings.BUCKET_CHECKPOINTS, Key=obj["Key"])
                    pending.append(json.loads(body["Body"].read()))
                except Exception:
                    logger.warning(f"Could not read pending marker {obj['Key']}")
    except Exception:
        logger.exception("Could not list pending jobs")
    return pending


def clear_pending(s3, job_id: str):
    try:
        s3.delete_object(Bucket=settings.BUCKET_CHECKPOINTS, Key=_pending_marker_key(job_id))
    except Exception:
        logger.exception(f"Could not clear pending marker for job {job_id}")


# ── K8s Job spawn + status ───────────────────────────────────────────────────

def spawn_k8s_training_job(job_id: str, level: str, checkpoint_key: str, terrain: str):
    with open(_K8S_JOB_YAML) as f:
        yaml_content = (
            f.read()
            .replace("JOB_ID", job_id)
            .replace("TRAINING_LEVEL_PLACEHOLDER", level)
            .replace("CHECKPOINT_KEY_PLACEHOLDER", checkpoint_key)
            .replace("TRAINING_TERRAIN_PLACEHOLDER", terrain)
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", tmp_path], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"kubectl apply failed: {result.stderr}")
        logger.info(f"K8s Job {job_id} created")
    finally:
        os.unlink(tmp_path)


def check_k8s_job_status(job_id: str) -> str:
    """
    Single, non-blocking check of a K8s Job's status — "Complete", "Failed",
    or "Running" (covers Pending/Running/unknown). Unlike the old
    wait_for_k8s_job(), this does not loop or sleep; the watcher calls this
    once per poll cycle and just checks again next cycle if still running.
    """
    result = subprocess.run(
        ["kubectl", "get", "job", job_id, "-o", "jsonpath={.status.conditions[0].type}"],
        capture_output=True, text=True,
    )
    status = result.stdout.strip()
    if status in ("Complete", "Failed"):
        return status
    return "Running"


# ── MLflow lookup + promotion ────────────────────────────────────────────────

def _get_all_experiment_ids() -> list[str]:
    try:
        experiments = client.search_experiments()
        ids = [e.experiment_id for e in experiments]
        logger.info(f"Querying across {len(ids)} MLflow experiment(s): {ids}")
        return ids
    except Exception as e:
        logger.warning(f"Could not list experiments ({e}) — falling back to experiment '0'")
        return ["0"]


def find_mlflow_run_once(job_id: str):
    """
    Single (non-looping) search for a FINISHED run tagged with this job_id.
    The watcher calls this once per poll cycle instead of blocking in a
    retry loop — if no run is found yet, it just checks again next cycle.
    """
    try:
        runs = client.search_runs(
            experiment_ids=_get_all_experiment_ids(),
            filter_string=f"tags.job_id = '{job_id}' and status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs[0] if runs else None
    except Exception:
        logger.exception(f"MLflow query error for job_id={job_id}")
        return None


def validate_and_promote(run, terrain: str) -> bool:
    """
    Validate accuracy + latency thresholds, then promote to Staging -> Production.
    Identical logic to the original model_trainer.py — unchanged on purpose.
    """
    accuracy   = run.data.metrics.get("accuracy", 0)
    latency    = run.data.metrics.get("latency_p95", 999)
    model_name = f"uav-navigator-{terrain}"
    logger.info(f"Metrics → accuracy={accuracy:.4f}  latency={latency:.1f}ms  model={model_name}")

    if accuracy < settings.MIN_ACCURACY:
        logger.warning(f"FAIL: accuracy {accuracy:.4f} < {settings.MIN_ACCURACY} — not promoting")
        return False
    if latency > settings.MAX_LATENCY:
        logger.warning(f"FAIL: latency {latency:.1f}ms > {settings.MAX_LATENCY}ms — not promoting")
        return False

    mock_tag = run.data.tags.get("mock_training", "false")
    if mock_tag == "true":
        logger.warning("Mock training run — promoting to Staging only, not Production")
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version))[-1]
                client.transition_model_version_stage(
                    name=model_name, version=latest.version, stage="Staging"
                )
                logger.info(f"{model_name} v{latest.version} → Staging (mock run)")
        except Exception:
            logger.exception("MLflow Staging promote error")
        return True

    logger.info(f"PASS — promoting {model_name} to Production")
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            client.transition_model_version_stage(
                name=model_name, version=latest.version, stage="Staging"
            )
            # archive_existing_versions=True: without this, MLflow does NOT
            # auto-demote the previous Production version — multiple versions
            # would silently accumulate in the Production stage across retrain
            # cycles, which breaks the "exactly one current Production version"
            # assumption consumers/rollback_model.py relies on.
            client.transition_model_version_stage(
                name=model_name, version=latest.version, stage="Production",
                archive_existing_versions=True,
            )
            logger.info(f"{model_name} v{latest.version} → Production")
    except Exception:
        logger.exception("MLflow Production promote error")

    return True
