import os

# ── Storage (Ceph S3) ──────────────────────────────────────────────────────
S3_ENDPOINT   = os.environ.get("S3_ENDPOINT",   "http://localhost:7480")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "uavaccess")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "uavsecret123")
S3_REGION     = os.environ.get("S3_REGION",     "us-east-1")

# ── Kafka ──────────────────────────────────────────────────────────────────
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")

KAFKA_TOPIC_RETRAIN = os.environ.get("KAFKA_TOPIC_RETRAIN", "uav-retrain")

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_URI         = os.environ.get("MLFLOW_TRACKING_URI",    "http://localhost:5000")
MLFLOW_S3_ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", S3_ENDPOINT)

# ── Bucket names ───────────────────────────────────────────────────────────
BUCKET_CHECKPOINTS   = "checkpoints"
BUCKET_TRAINING_DATA = "training-data"      
BUCKET_FAST_WEIGHT   = "fast-weight-state"  

# ── Nested Learning thresholds ─────────────────────────────────────────────
# Thresholds compare against Titans surprise score (MSE between two unit vectors).
#
# IMPORTANT: both recall r and input x are L2-normalised to unit sphere before MSE.
# For unit vectors in R^512: MSE = ||r - x||² / 512 ∈ [0, 4/512] ≈ [0, 0.0078]
#   MSE ≈ 0.0000 → terrain fully familiar (r ≈ x)
#   MSE ≈ 0.0020 → mild change (W=0 cold-start baseline)
#   MSE ≈ 0.0078 → maximum possible (r and x antiparallel)
#
# Old Hebbian thresholds were cosine-distance [0,2]; divide by ~256 to convert:
#   0.30 → 0.0010  |  0.80 → 0.0030  |  50.0 → 0.10
# (accumulator calibrated for 50-frame demo: avg_surprise ~0.002 × ~50 frames ≈ 0.10)
NL_FAST_DELTA       = float(os.environ.get("NL_FAST_DELTA",       "0.0010"))
NL_MEDIUM_DELTA     = float(os.environ.get("NL_MEDIUM_DELTA",     "0.0030"))
NL_SLOW_ACCUMULATOR = float(os.environ.get("NL_SLOW_ACCUMULATOR", "0.10"))

# ── Titans memory (replaces Hebbian CMS) ──────────────────────────────────
# W matrices: same embed dim, same 3 timescales, same blend weights as before.
# Forgetting factors (NL_ALPHA_*) unchanged — control how fast each scale forgets.
# TITANS_LR_*: gradient step size for each timescale.
#   Surprise score (MSE of two unit vectors in R^512) is bounded [0, 4/512] ≈ [0, 0.008]:
#     ~0.000 = terrain fully familiar  |  ~0.002 = cold-start  |  ~0.008 = completely new
# NL_*_DELTA thresholds are re-scaled vs Hebbian cosine-distance defaults:
#   Hebbian cosine-dist [0,2]; Titans MSE [0,4] → thresholds roughly doubled.
NL_EMBED_DIM    = int(os.environ.get("NL_EMBED_DIM",    "512"))
NL_ALPHA_FAST   = float(os.environ.get("NL_ALPHA_FAST",   "0.90"))
NL_ALPHA_MEDIUM = float(os.environ.get("NL_ALPHA_MEDIUM", "0.95"))
NL_ALPHA_SLOW_W = float(os.environ.get("NL_ALPHA_SLOW_W", "0.99"))
NL_BLEND_FAST   = float(os.environ.get("NL_BLEND_FAST",   "0.30"))
NL_BLEND_MEDIUM = float(os.environ.get("NL_BLEND_MEDIUM", "0.20"))
NL_BLEND_SLOW_W = float(os.environ.get("NL_BLEND_SLOW_W", "0.10"))

# Titans learning rates — how aggressively each timescale updates per frame
TITANS_LR_FAST  = float(os.environ.get("TITANS_LR_FAST",  "0.05"))
TITANS_LR_MED   = float(os.environ.get("TITANS_LR_MED",   "0.01"))
TITANS_LR_SLOW  = float(os.environ.get("TITANS_LR_SLOW",  "0.002"))

# ── Model validation thresholds ────────────────────────────────────────────
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.75"))
MAX_LATENCY  = float(os.environ.get("MAX_LATENCY",  "200.0"))

# ── Simulator ──────────────────────────────────────────────────────────────
FRAME_DELAY = float(os.environ.get("FRAME_DELAY", "0.5"))
FLIGHT_TERRAIN = os.environ.get("FLIGHT_TERRAIN", "unknown")
