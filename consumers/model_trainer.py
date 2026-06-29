import json
import logging
import os
import subprocess
import tempfile
import time

import mlflow
from mlflow import MlflowClient

from config import settings
from infra.kafka_consumer import make_consumer
from infra.s3_client import make_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [model_trainer] %(levelname)s %(message)s",
)
logger = logging.getLogger("model_trainer")

# MLflow env setup
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]      = settings.S3_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]  = settings.S3_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"]     = settings.S3_REGION

mlflow.set_tracking_uri(settings.MLFLOW_URI)
client = MlflowClient()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K8S_JOB_YAML = os.path.join(_PROJECT_ROOT, "k8s", "training_job.yaml")

# ── Kafka deduplication ────────────────────────────────────────────────────
# In-process set of checkpoint keys already processed this session.
# Prevents duplicate K8s jobs when flight_agent retries a Kafka publish.
# Note: this resets on consumer restart — for cross-restart dedup, persist to Redis/DB.
_seen_keys: set[str] = set()


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


def wait_for_k8s_job(job_id: str, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["kubectl", "get", "job", job_id,
             "-o", "jsonpath={.status.conditions[0].type}"],
            capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status == "Complete":
            logger.info(f"K8s Job {job_id} completed")
            return True
        if status == "Failed":
            logger.error(f"K8s Job {job_id} failed")
            return False
        time.sleep(10)
    logger.error(f"K8s Job {job_id} timed out")
    return False


def _get_all_experiment_ids() -> list[str]:
    """
    Return all MLflow experiment IDs.

    Previously hardcoded to ["0"] (default experiment only), which silently
    missed runs logged to non-default experiments. Now queries all active experiments.
    """
    try:
        experiments = client.search_experiments()
        ids = [e.experiment_id for e in experiments]
        logger.info(f"Querying across {len(ids)} MLflow experiment(s): {ids}")
        return ids
    except Exception as e:
        logger.warning(f"Could not list experiments ({e}) — falling back to experiment '0'")
        return ["0"]


def wait_for_mlflow_run(job_id: str, timeout: int = 300):
    """
    Wait for a FINISHED MLflow run tagged with this exact job_id.
    Filters by tags.job_id so parallel consumers never steal each other's run.
    """
    deadline       = time.time() + timeout
    experiment_ids = _get_all_experiment_ids()

    while time.time() < deadline:
        try:
            runs = client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=f"tags.job_id = '{job_id}' and status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                return runs[0]
        except Exception:
            logger.exception("MLflow query error")
        time.sleep(5)

    return None


def validate_and_promote(run, terrain: str) -> bool:
    """
    Validate accuracy + latency thresholds, then promote to Staging → Production.

    Stage order: None → Staging → Production (MLflow requires sequential promotion).
    feature_extractor.py prefers Production > Staging, so only promoted models
    will be loaded by the flight agent on the next flight.
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

    # Check if this was a mock training run — don't promote mock metrics to Production
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
            # MLflow requires Staging before Production
            client.transition_model_version_stage(
                name=model_name, version=latest.version, stage="Staging"
            )
            client.transition_model_version_stage(
                name=model_name, version=latest.version, stage="Production"
            )
            logger.info(f"{model_name} v{latest.version} → Production")
    except Exception:
        logger.exception("MLflow Production promote error")

    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    s3       = make_s3_client()
    consumer = make_consumer(topic=settings.KAFKA_TOPIC_RETRAIN, group_id="model-trainer-group")
    logger.info(f"Ready — listening on topic={settings.KAFKA_TOPIC_RETRAIN}")

    for msg in consumer:
        event = msg.value

        if event.get("event") != "retrain_trigger":
            logger.debug(f"Skipping non-retrain event: {event.get('event')}")
            continue

        key          = event.get("checkpoint_key", "")
        level        = event.get("level", "slow")
        triggered_at = event.get("timestamp", time.time())

        # ── Deduplication ─────────────────────────────────────────────────
        if key in _seen_keys:
            logger.info(f"Duplicate event for key={key} — skipping (already processed this session)")
            continue
        _seen_keys.add(key)

        logger.info(f"Retrain event received: level={level}  key={key}")

        # ── Read training metadata from Ceph ──────────────────────────────
        try:
            obj             = s3.get_object(Bucket=settings.BUCKET_CHECKPOINTS, Key=key)
            checkpoint_info = json.loads(obj["Body"].read())
            logger.info(f"Checkpoint info: {checkpoint_info}")
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

        if not wait_for_k8s_job(job_id):
            continue

        logger.info(f"Querying MLflow for run tagged job_id={job_id}...")
        new_run = wait_for_mlflow_run(job_id, timeout=300)
        if not new_run:
            logger.error("No new MLflow run found (timeout)")
            continue

        logger.info(f"Found MLflow run: {new_run.info.run_id}")
        validate_and_promote(new_run, terrain)


if __name__ == "__main__":
    main()
