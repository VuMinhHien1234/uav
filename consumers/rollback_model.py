"""
consumers/rollback_model.py — manually roll back a terrain's Production
model to a previous version on MLflow.

There is no automatic rollback anywhere in this pipeline: validate_and_promote()
(consumers/_trainer_common.py) only checks accuracy/latency on the training
set right after training finishes — it has no visibility into how a model
actually performs once it's flying. If a promoted model turns out to be
worse in real flights, use this script to move Production back.

Usage:
    python3 -m consumers.rollback_model --terrain forest --list
        # show every version of uav-navigator-forest and its current stage

    python3 -m consumers.rollback_model --terrain forest
        # roll back Production to the version immediately before the
        # current one

    python3 -m consumers.rollback_model --terrain forest --version 4
        # roll back Production to a specific version number
"""
import argparse
import logging

import mlflow
from mlflow import MlflowClient

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rollback] %(levelname)s %(message)s",
)
logger = logging.getLogger("rollback")

mlflow.set_tracking_uri(settings.MLFLOW_URI)
client = MlflowClient()


def list_versions(model_name: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"No versions found for '{model_name}'")
        return []
    versions = sorted(versions, key=lambda v: int(v.version))
    for v in versions:
        print(f"v{v.version:<4} stage={v.current_stage:<10} run_id={v.run_id}")
    return versions


def _current_production_version(model_name: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    prod = [v for v in versions if v.current_stage == "Production"]
    return sorted(prod, key=lambda v: int(v.version))[-1] if prod else None


def rollback(terrain: str, target_version: str = None) -> None:
    model_name = f"uav-navigator-{terrain}"
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise SystemExit(f"No versions found for model '{model_name}'")
    versions = sorted(versions, key=lambda v: int(v.version))

    current = _current_production_version(model_name)
    if current is None:
        raise SystemExit(
            f"No version currently in Production for '{model_name}' — nothing to roll back from"
        )

    if target_version is None:
        # Default: the version immediately before the current Production one.
        older = [v for v in versions if int(v.version) < int(current.version)]
        if not older:
            raise SystemExit(
                f"v{current.version} is the earliest version of '{model_name}' — "
                f"no older version to roll back to"
            )
        target = older[-1]
    else:
        matches = [v for v in versions if v.version == str(target_version)]
        if not matches:
            raise SystemExit(f"Version {target_version} not found for '{model_name}'")
        target = matches[0]

    if target.version == current.version:
        raise SystemExit(f"v{target.version} is already the current Production version — nothing to do")

    logger.info(
        f"Rolling back {model_name}: v{current.version} (Production) -> v{target.version}"
    )

    # Demote the current Production version so exactly one version holds
    # Production at a time (matches how validate_and_promote() operates).
    client.transition_model_version_stage(name=model_name, version=current.version, stage="Archived")

    # MLflow requires passing through Staging before Production if the target
    # isn't already at Staging/Production.
    if target.current_stage not in ("Staging", "Production"):
        client.transition_model_version_stage(name=model_name, version=target.version, stage="Staging")
    client.transition_model_version_stage(name=model_name, version=target.version, stage="Production")

    logger.info(f"Done: {model_name} v{target.version} is now Production (was v{current.version})")


def main():
    parser = argparse.ArgumentParser(
        description="Roll back a terrain's Production model on MLflow"
    )
    parser.add_argument("--terrain", required=True, help="Terrain name, e.g. forest")
    parser.add_argument(
        "--version", default=None,
        help="Specific version to roll back to (default: the version before current Production)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all versions + stages for this terrain, then exit (no changes made)",
    )
    args = parser.parse_args()

    model_name = f"uav-navigator-{args.terrain}"

    if args.list:
        list_versions(model_name)
        return

    rollback(args.terrain, args.version)


if __name__ == "__main__":
    main()
