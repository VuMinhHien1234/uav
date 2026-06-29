#!/bin/bash
# ============================================================
# UAV Nested Learning Pipeline — Full Demo Script
# Run from project root after completing setup
# ============================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}"

echo "============================================================"
echo " UAV Nested Learning Demo"
echo "============================================================"
echo ""

# ── Verify services ──────────────────────────────────────────────────────────
echo "[0] Checking services..."
aws --endpoint-url http://localhost:7480 s3 ls > /dev/null 2>&1 \
  && echo "  ✓ Ceph RadosGW OK" \
  || { echo "  ✗ Ceph NOT running. Check: curl http://localhost:7480"; exit 1; }

curl -sf http://localhost:5000/health > /dev/null \
  && echo "  ✓ MLflow OK" \
  || { echo "  ✗ MLflow NOT running"; exit 1; }

docker exec kafka kafka-topics --list --bootstrap-server localhost:9092 > /dev/null 2>&1 \
  && echo "  ✓ Kafka OK" \
  || { echo "  ✗ Kafka NOT running"; exit 1; }

kubectl get nodes > /dev/null 2>&1 \
  && echo "  ✓ K8s (k3s) OK" \
  || { echo "  ✗ k3s NOT running. Run: sudo systemctl start k3s"; exit 1; }

sudo k3s ctr images ls 2>/dev/null | grep -q uav-trainer \
  && echo "  ✓ uav-trainer image OK" \
  || { echo "  ✗ uav-trainer image missing. Run Bước 9 from SETUP.md:"; \
       echo "      docker build -t uav-trainer:latest ."; \
       echo "      docker tag uav-trainer:latest localhost:5001/uav-trainer:latest"; \
       echo "      docker push localhost:5001/uav-trainer:latest"; \
       echo "      sudo k3s ctr images pull --plain-http localhost:5001/uav-trainer:latest"; \
       exit 1; }
echo ""

# ── Generate mock data ────────────────────────────────────────────────────────
echo "[1] Generating mock frames..."
python3 scripts/download_mock_data.py
echo ""

# ── Start consumers ───────────────────────────────────────────────────────────
echo "[2] Starting consumers (background)..."
python3 -m consumers.model_trainer > /tmp/model_trainer.log 2>&1 &
UC2_PID=$!
echo "  model_trainer PID: $UC2_PID  (tail /tmp/model_trainer.log)"

echo "  Waiting 5s for consumer to connect..."
sleep 5
echo ""

# ── Run flight_agent — one terrain at a time ─────────────────────────────────
# Each terrain trains a separate model: uav-navigator-{terrain}
# To demo a specific terrain:  FLIGHT_TERRAIN=mountain bash scripts/demo.sh
#
# Available terrains: urban | forest | desert | coastal | mountain

TERRAIN="${FLIGHT_TERRAIN:-mountain}"

echo "[3] Starting flight_agent | terrain=${TERRAIN}"
echo "    Model will be registered as: uav-navigator-${TERRAIN}"
echo "    (change terrain: FLIGHT_TERRAIN=forest bash scripts/demo.sh)"
echo "------------------------------------------------------------"
FLIGHT_TERRAIN="${TERRAIN}" python3 -m simulator.flight_agent
echo "------------------------------------------------------------"
echo ""

# ── Check results ─────────────────────────────────────────────────────────────
echo "[4] Results..."

echo ""
echo "  --- Ceph Buckets ---"
for bucket in checkpoints training-data mlflow-artifacts fast-weight-state; do
  count=$(aws --endpoint-url http://localhost:7480 s3 ls s3://$bucket/ --recursive 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-22s %s objects\n" "$bucket:" "$count"
done

echo ""
echo "  --- MLflow ---"
python3 - <<'PYEOF'
import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
try:
    exp_ids = [e.experiment_id for e in client.search_experiments()]
except Exception:
    exp_ids = ["0"]
runs = mlflow.search_runs(experiment_ids=exp_ids, order_by=["start_time DESC"], max_results=3)
if runs.empty:
    print("  No runs yet (model_trainer may still be processing)")
else:
    for _, r in runs.iterrows():
        print(f"  Run: {r['run_id'][:8]}...  acc={r.get('metrics.accuracy','?')}  lat={r.get('metrics.latency_p95','?')}ms")
PYEOF

echo ""
echo "  --- Consumer logs (last 5 lines) ---"
echo "  [model_trainer]:"
tail -5 /tmp/model_trainer.log 2>/dev/null | sed 's/^/    /'

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
echo "[5] Stopping consumers..."
kill $UC2_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo " Demo complete!"
echo " MLflow UI:        http://localhost:5000"
echo " Ceph dashboard:   https://localhost:8443  (admin/adminpassword)"
echo "============================================================"
