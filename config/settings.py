"""
Centralized configuration — reads from environment variables with sensible defaults.
Override any value by setting the corresponding env var before running.
"""
import os

# ── Storage (Ceph S3) ──────────────────────────────────────────────────────
S3_ENDPOINT   = os.environ.get("S3_ENDPOINT",   "http://localhost:7480")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "uavaccess")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "uavsecret123")
S3_REGION     = os.environ.get("S3_REGION",     "us-east-1")

# ── Kafka ──────────────────────────────────────────────────────────────────
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC  = os.environ.get("KAFKA_TOPIC",  "ai-pipeline")

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_URI         = os.environ.get("MLFLOW_TRACKING_URI",    "http://localhost:5000")
MLFLOW_S3_ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", S3_ENDPOINT)

# ── Bucket names ───────────────────────────────────────────────────────────
BUCKET_RAW_FRAMES    = "raw-frames"
BUCKET_EMBEDDINGS    = "embeddings"
BUCKET_CHECKPOINTS   = "checkpoints"
BUCKET_TRAINING_DATA = "training-data"
BUCKET_MLFLOW        = "mlflow-artifacts"

# ── Nested Learning thresholds ─────────────────────────────────────────────
NL_MEMORY_THRESHOLD = float(os.environ.get("NL_MEMORY_THRESHOLD", "0.85"))
NL_MEDIUM_DELTA     = float(os.environ.get("NL_MEDIUM_DELTA",     "0.40"))
NL_FAST_DELTA       = float(os.environ.get("NL_FAST_DELTA",       "0.15"))
NL_SLOW_ACCUMULATOR = float(os.environ.get("NL_SLOW_ACCUMULATOR", "50.0"))

# ── Model validation thresholds ────────────────────────────────────────────
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.75"))
MAX_LATENCY  = float(os.environ.get("MAX_LATENCY",  "200.0"))

# ── Simulator ──────────────────────────────────────────────────────────────
FRAME_DELAY = float(os.environ.get("FRAME_DELAY", "0.5"))
