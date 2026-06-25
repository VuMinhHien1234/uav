"""
UC1 Consumer — Frame Preprocessing + Path Memory Builder

Trigger : Kafka event when a new frame lands in Ceph raw-frames/
Action  : Extract embedding (ResNet18) → save to embeddings/ (path memory)
          Copy frame to training-data/ for slow retrain
"""
import json
import logging
import time

from config import settings
from core.feature_extractor import FeatureExtractor
from infra.kafka_consumer import make_consumer
from infra.s3_client import make_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [uc1] %(levelname)s %(message)s",
)
logger = logging.getLogger("uc1")


def _save_path_memory(s3, frame_id: str, embedding: list, source_key: str):
    entry = {
        "frame_id":   frame_id,
        "embedding":  embedding,
        "dims":       len(embedding),
        "timestamp":  time.time(),
        "source_key": source_key,
    }
    s3.put_object(
        Bucket=settings.BUCKET_EMBEDDINGS,
        Key=f"path-memory/{frame_id}.json",
        Body=json.dumps(entry),
        ContentType="application/json",
    )


def main():
    s3        = make_s3_client()
    extractor = FeatureExtractor()
    consumer  = make_consumer(group_id="uc1-consumer")

    logger.info("Ready — waiting for frames in raw-frames/")

    for msg in consumer:
        for record in msg.value.get("Records", []):
            bucket = record["s3"]["bucket"]["name"]
            key    = record["s3"]["object"]["key"]

            if bucket != settings.BUCKET_RAW_FRAMES:
                continue

            logger.info(f"Received: {key}")
            try:
                img_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                logger.debug(f"Size: {len(img_bytes)} bytes")

                embedding = extractor.from_bytes(img_bytes)
                logger.debug(f"Embedding: {len(embedding)} dims  [{embedding[0]:.3f}, ...]")

                frame_id = key.replace("/", "_").replace(".jpg", "").replace(".png", "")
                _save_path_memory(s3, frame_id, embedding, key)
                logger.info(f"Path memory → embeddings/path-memory/{frame_id}.json")

                s3.copy_object(
                    CopySource={"Bucket": bucket, "Key": key},
                    Bucket=settings.BUCKET_TRAINING_DATA,
                    Key=f"frames/{key}",
                )
                logger.info(f"Copied → training-data/frames/{key}")

            except Exception:
                logger.exception(f"Failed to process {key}")


if __name__ == "__main__":
    main()
