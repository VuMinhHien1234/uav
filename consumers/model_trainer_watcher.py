"""
model_trainer_watcher.py — polls K8s Jobs spawned by model_trainer.py to
completion, then validates + promotes the resulting MLflow model.

Runs as a separate, independent process from model_trainer.py's consumer
loop. The two only ever talk through Ceph markers under the `checkpoints`
bucket (pending/<job_id>.json, processed/<checkpoint_key>) — no shared
memory, no direct RPC — so either process can be restarted independently
without losing track of in-flight jobs: on restart, this watcher just
re-lists pending/*.json and picks up exactly where it left off.

Run alongside model_trainer.py:
    python3 -m consumers.model_trainer_watcher
"""
import logging
import time

from core.titans_aggregate import aggregate_all_terrains
from infra.s3_client import make_s3_client

from consumers._trainer_common import (
    check_k8s_job_status,
    clear_pending,
    find_mlflow_run_once,
    list_pending_jobs,
    mark_processed,
    validate_and_promote,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [model_trainer_watcher] %(levelname)s %(message)s",
)
logger = logging.getLogger("model_trainer_watcher")

POLL_INTERVAL_SEC   = 15   # how often to re-check every pending job
JOB_TIMEOUT_SEC     = 600  # matches the old wait_for_k8s_job() timeout
MLFLOW_TIMEOUT_SEC  = 300  # matches the old wait_for_mlflow_run() timeout
AGGREGATE_EVERY_N   = 4    # run Titans aggregation every 4th poll cycle (~60s)


def _handle_job(s3, record: dict):
    job_id         = record["job_id"]
    checkpoint_key = record["checkpoint_key"]
    terrain        = record["terrain"]
    spawned_at     = record.get("spawned_at", time.time())
    age            = time.time() - spawned_at

    status = check_k8s_job_status(job_id)

    if status == "Running":
        if age > JOB_TIMEOUT_SEC:
            logger.error(f"Job {job_id} timed out after {JOB_TIMEOUT_SEC}s — giving up")
            mark_processed(s3, checkpoint_key, status="job_timeout")
            clear_pending(s3, job_id)
        return  # still running — check again next poll cycle

    if status == "Failed":
        logger.error(f"Job {job_id} failed")
        mark_processed(s3, checkpoint_key, status="job_failed")
        clear_pending(s3, job_id)
        return

    # status == "Complete"
    run = find_mlflow_run_once(job_id)
    if run is None:
        if age > JOB_TIMEOUT_SEC + MLFLOW_TIMEOUT_SEC:
            logger.error(f"No MLflow run found for job_id={job_id} after waiting — giving up")
            mark_processed(s3, checkpoint_key, status="mlflow_run_not_found")
            clear_pending(s3, job_id)
        return  # MLflow run not visible yet — retry next poll cycle

    logger.info(f"Job {job_id} complete — found MLflow run {run.info.run_id}")
    validate_and_promote(run, terrain)
    mark_processed(s3, checkpoint_key, status="done")
    clear_pending(s3, job_id)


def main():
    s3 = make_s3_client()
    logger.info(f"Watcher ready — polling every {POLL_INTERVAL_SEC}s")
    cycle = 0
    while True:
        pending = list_pending_jobs(s3)
        if pending:
            logger.info(f"{len(pending)} pending job(s) to check")
        for record in pending:
            try:
                _handle_job(s3, record)
            except Exception:
                logger.exception(f"Error handling pending job {record.get('job_id')}")

        # Merge per-flight Titans states into latest.pt every few cycles —
        # see core/titans_aggregate.py for why this replaces the old
        # last-write-wins overwrite of latest.pt.
        if cycle % AGGREGATE_EVERY_N == 0:
            try:
                aggregate_all_terrains(s3)
            except Exception:
                logger.exception("Titans aggregation cycle failed")

        cycle += 1
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
