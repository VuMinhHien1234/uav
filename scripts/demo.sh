#!/bin/bash
# ============================================================
# UAV Nested Learning Pipeline — Full Demo Script
# Run from project root after completing setup
# ============================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " UAV Nested Learning Demo"
echo "============================================================"
echo ""

# ── Verify services ──────────────────────────────────────────────────────────
echo "[0] Checking services..."
radosgw-admin bucket list > /dev/null 2>&1 \
  && echo "  ✓ Ceph RadosGW OK" \
  || { echo "  ✗ Ceph NOT running"; exit 1; }

curl -sf http://localhost:5000/health > /dev/null \
  && echo "  ✓ MLflow OK" \
  || { echo "  ✗ MLflow NOT running"; exit 1; }

kafka-topics.sh --list --bootstrap-server localhost:9092 > /dev/null 2>&1 \
  && echo "  ✓ Kafka OK" \
  || { echo "  ✗ Kafka NOT running"; exit 1; }

kubectl get nodes > /dev/null 2>&1 \
  && echo "  ✓ K8s (k3s) OK" \
  || { echo "  ✗ k3s NOT running. Run: sudo systemctl start k3s"; exit 1; }

sudo k3s ctr images ls 2>/dev/null | grep -q uav-trainer \
  && echo "  ✓ uav-trainer image OK" \
  || { echo "  ✗ uav-trainer image missing. Run: docker build -t uav-trainer:latest . && docker save uav-trainer:latest | sudo k3s ctr images import -"; exit 1; }
echo ""

# ── Generate mock data ────────────────────────────────────────────────────────
echo "[1] Generating mock frames..."
python3 scripts/download_mock_data.py
echo ""

# ── Start consumers ───────────────────────────────────────────────────────────
echo "[2] Starting consumers (background)..."
python3 -m consumers.uc1_frame_processor > /tmp/uc1.log 2>&1 &
UC1_PID=$!
echo "  UC1 PID: $UC1_PID  (tail /tmp/uc1.log)"

python3 -m consumers.uc2_checkpoint_validator > /tmp/uc2.log 2>&1 &
UC2_PID=$!
echo "  UC2 PID: $UC2_PID  (tail /tmp/uc2.log)"

echo "  Waiting 5s for consumers to connect..."
sleep 5
echo ""

# ── Run UC3 simulator ─────────────────────────────────────────────────────────
echo "[3] Starting UC3 Simulator (foreground)..."
echo "    Watching for: memory_hit | slow | medium | fast | skip"
echo "------------------------------------------------------------"
python3 -m simulator.uc3_simulator
echo "------------------------------------------------------------"
echo ""

# ── Check results ─────────────────────────────────────────────────────────────
echo "[4] Results..."

echo ""
echo "  --- Ceph Buckets ---"
for bucket in raw-frames embeddings checkpoints mlflow-artifacts; do
  count=$(aws --endpoint-url http://localhost:7480 s3 ls s3://$bucket/ --recursive 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-18s %s objects\n" "$bucket:" "$count"
done

echo ""
echo "  --- MLflow ---"
python3 - <<'PYEOF'
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=3)
if runs.empty:
    print("  No runs yet (UC2 may still be processing)")
else:
    for _, r in runs.iterrows():
        print(f"  Run: {r['run_id'][:8]}...  acc={r.get('metrics.accuracy','?')}  lat={r.get('metrics.latency_p95','?')}ms")
PYEOF

echo ""
echo "  --- KServe ---"
kubectl get inferenceservice uav-navigator 2>/dev/null || echo "  (not yet updated)"

echo ""
echo "  --- Consumer logs (last 5 lines each) ---"
echo "  [UC1]:"
tail -5 /tmp/uc1.log 2>/dev/null | sed 's/^/    /'
echo "  [UC2]:"
tail -5 /tmp/uc2.log 2>/dev/null | sed 's/^/    /'

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
echo "[5] Stopping consumers..."
kill $UC1_PID $UC2_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo " Demo complete!"
echo " MLflow UI:        http://localhost:5000"
echo " Ceph dashboard:   https://localhost:8443  (admin/adminpassword)"
echo "============================================================"
