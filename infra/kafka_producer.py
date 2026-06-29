"""KafkaProducer factory — flight_agent publishes retrain events to Kafka.

Production pattern: flight_agent writes slow/medium frames directly to
training-data/frames/terrain_{env}/flight_{id}/ in Ceph (storage), then publishes a
retrain_trigger event to Kafka (notification). model_trainer consumes the
event and spawns a K8s training job. No raw-frames bucket, no frame_indexer.
"""
import json
import logging

from kafka import KafkaProducer

from config import settings

logger = logging.getLogger(__name__)


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=[settings.KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",    # wait for all in-sync replicas before ack
        retries=3,
        linger_ms=10,  # small batch window to reduce round-trips
    )


def _publish(producer: KafkaProducer, topic: str, event: dict):
    """Send one event to the given topic and block until acked (max 5s)."""
    future = producer.send(topic, value=event)
    try:
        future.get(timeout=5)
    except Exception as e:
        logger.error(f"Kafka publish failed (topic={topic}): {e}")


def publish_retrain_event(
    producer: KafkaProducer,
    level: str,
    checkpoint_key: str,
    frame_id: str,
    timestamp: float,
):
    """
    Publish retrain_trigger to uav-retrain topic.
    model_trainer consumes this and spawns a K8s training job.

    Event schema:
        event          : "retrain_trigger"
        level          : "slow" | "medium"
        checkpoint_key : Ceph object key under BUCKET_CHECKPOINTS
        frame_id       : frame that triggered the event
        timestamp      : unix epoch float
    """
    event = {
        "event":          "retrain_trigger",
        "level":          level,
        "checkpoint_key": checkpoint_key,
        "frame_id":       frame_id,
        "timestamp":      timestamp,
    }
    _publish(producer, settings.KAFKA_TOPIC_RETRAIN, event)
    logger.info(f"Kafka → {settings.KAFKA_TOPIC_RETRAIN}  level={level}  key={checkpoint_key}")
