"""
UC2 Consumer — Checkpoint Validator + MLflow Model Lifecycle Manager

Trigger : Kafka event when UC3 uploads a checkpoint to Ceph checkpoints/
Action  : Spawn K8s Training Job → wait for completion → validate MLflow metrics
          → transition model stage → update KServe InferenceService
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
from infra.kafka_consumer import make_consumer
from infra.s3_client import make_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [uc2] %(levelname)s %(message)s",
)
logger = logging.getLogger("uc2")

# MLflow env setup
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]      = settings.S3_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]  = settings.S3_SECRET_KEY
os.environ["AWS_DEFAULT_REGION"]     = settings.S3_REGION

mlflow.set_tracking_uri(settings.MLFLOW_URI)
client = MlflowClient()


# ── K8s helpers ───────────────────────────────────────────────────────────────

def spawn_k8s_training_job(job_id: str, level: str, checkpoint_key: str):
    with open("k8s/training_job.yaml") as f:
        yaml_content = (
            f.read()
            .replace("JOB_ID", job_id)
            .replace("TRAINING_LEVEL_PLACEHOLDER", level)
            .replace("CHECKPOINT_KEY_PLACEHOLDER", checkpoint_key)
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


# ── MLflow helpers ────────────────────────────────────────────────────────────

def wait_for_mlflow_run(triggered_at: float, timeout: int = 300):
    """Wait for a new FINISHED MLflow run that started after triggered_at."""
    triggered_ms = int(triggered_at * 1000)
    deadline     = time.time() + timeout

    while time.time() < deadline:
        try:
            runs = client.search_runs(
                experiment_ids=["0"],
                filter_string=(
                    f"status = 'FINISHED' "
                    f"and attributes.start_time > {triggered_ms}"
                ),
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                return runs[0]
        except Exception:
            logger.exception("MLflow query error")
        time.sleep(5)

    return None


def validate_and_promote(run) -> bool:
    accuracy = run.data.metrics.get("accuracy", 0)
    latency  = run.data.metrics.get("latency_p95", 999)
    logger.info(f"Metrics → accuracy={accuracy:.4f}  latency={latency:.1f}ms")

    if accuracy < settings.MIN_ACCURACY:
        logger.warning(f"FAIL: accuracy {accuracy:.4f} < {settings.MIN_ACCURACY}")
        return False
    if latency > settings.MAX_LATENCY:
        logger.warning(f"FAIL: latency {latency:.1f}ms > {settings.MAX_LATENCY}ms")
        return False

    logger.info("PASS — promoting model to Staging")

    try:
        versions = client.search_model_versions("name='uav-navigator'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            client.transition_model_version_stage(
                name="uav-navigator", version=latest.version, stage="Staging"
            )
            logger.info(f"Model v{latest.version} → Staging")
            _update_kserve(f"{run.info.artifact_uri}/uav-model")
    except Exception:
        logger.exception("MLflow promote error")

    return True


def _update_kserve(artifact_uri: str):
    patch      = {"spec": {"predictor": {"model": {"storageUri": artifact_uri}}}}
    patch_path = "/tmp/kserve-patch.json"
    with open(patch_path, "w") as f:
        json.dump(patch, f)

    result = subprocess.run(
        ["kubectl", "patch", "inferenceservice", "uav-navigator",
         "--type=merge", "--patch-file", patch_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("KServe InferenceService updated")
    else:
        logger.error(f"KServe patch failed: {result.stderr}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    s3       = make_s3_client()
    consumer = make_consumer(group_id="uc2-consumer")
    logger.info("Ready — waiting for checkpoint events in checkpoints/")

    for msg in consumer:
        for record in msg.value.get("Records", []):
            bucket = record["s3"]["bucket"]["name"]
            key    = record["s3"]["object"]["key"]

            if bucket != settings.BUCKET_CHECKPOINTS:
                continue

            logger.info(f"Received checkpoint: {key}")
            triggered_at = time.time()
            level = key.split("/")[0] if "/" in key else "slow"

            try:
                obj             = s3.get_object(Bucket=settings.BUCKET_CHECKPOINTS, Key=key)
                checkpoint_info = json.loads(obj["Body"].read())
                logger.info(f"Checkpoint info: {checkpoint_info}")
            except Exception:
                logger.exception(f"Cannot read checkpoint {key}")
                continue

            job_id = f"uav-train-{int(time.time())}"
            try:
                spawn_k8s_training_job(job_id, level, key)
            except Exception:
                logger.exception("Failed to spawn K8s Job")
                continue

            if not wait_for_k8s_job(job_id):
                continue

            logger.info(f"Querying MLflow for new run after {triggered_at:.0f}...")
            new_run = wait_for_mlflow_run(triggered_at, timeout=300)
            if not new_run:
                logger.error("No new MLflow run found (timeout)")
                continue

            logger.info(f"Found MLflow run: {new_run.info.run_id}")
            validate_and_promote(new_run)


if __name__ == "__main__":
    main()
