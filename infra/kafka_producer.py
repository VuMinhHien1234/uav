"""KafkaProducer factory — flight_agent publishes retrain events to Kafka.

Production pattern: flight_agent writes slow/medium frames directly to
training-data/frames/terrain_{env}/flight_{id}/ in Ceph (storage), then publishes a
retrain_trigger event to Kafka (notification). model_trainer consumes the
event and spawns a K8s training job. No raw-frames bucket, no frame_indexer.
"""
import json
import logging
import os

from kafka import KafkaProducer

from config import settings

logger = logging.getLogger(__name__)

# ── Local durability buffer ─────────────────────────────────────────────────
# If Kafka is unreachable when flight_agent tries to publish a retrain_trigger,
# the checkpoint/frame is already safely written to Ceph — but the "go retrain
# this" signal must not just vanish. Failed publishes are appended here (one
# JSON object per line) and retried opportunistically on the next call to
# _publish(), so a Kafka outage delays training instead of silently losing it.
_BUFFER_PATH = os.environ.get(
    "KAFKA_PENDING_BUFFER_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "kafka_pending_events.jsonl",
    ),
)


def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=[settings.KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",    # wait for all in-sync replicas before ack
        retries=3,
        linger_ms=10,  # small batch window to reduce round-trips
    )


def _append_to_buffer(topic: str, event: dict):
    """Persist an event we failed to publish so a Kafka outage doesn't lose it."""
    try:
        with open(_BUFFER_PATH, "a") as f:
            f.write(json.dumps({"topic": topic, "event": event}) + "\n")
        logger.warning(f"Buffered undelivered event to {_BUFFER_PATH} (topic={topic})")
    except Exception:
        logger.exception(f"Could not write to local Kafka buffer {_BUFFER_PATH} — event lost")


def flush_pending_events(producer: KafkaProducer) -> int:
    """
    Best-effort retry of events buffered from previous publish failures.

    Reads _BUFFER_PATH, tries to resend each line. Lines that fail again are
    kept (in order) for the next attempt; lines that succeed are dropped.
    Cheap no-op when the buffer doesn't exist, so safe to call before every
    publish (see _publish() below) as well as from a standalone retry loop.

    Returns the number of events successfully flushed.
    """
    if not os.path.exists(_BUFFER_PATH):
        return 0

    try:
        with open(_BUFFER_PATH, "r") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except Exception:
        logger.exception(f"Could not read Kafka buffer {_BUFFER_PATH}")
        return 0

    if not lines:
        return 0

    still_pending = []
    flushed = 0
    for line in lines:
        try:
            record = json.loads(line)
            future = producer.send(record["topic"], value=record["event"])
            future.get(timeout=5)
            flushed += 1
        except Exception:
            still_pending.append(line)

    try:
        if still_pending:
            with open(_BUFFER_PATH, "w") as f:
                f.write("\n".join(still_pending) + "\n")
        else:
            os.remove(_BUFFER_PATH)
    except Exception:
        logger.exception(f"Could not rewrite Kafka buffer {_BUFFER_PATH}")

    if flushed:
        logger.info(
            f"Flushed {flushed} previously buffered Kafka event(s); "
            f"{len(still_pending)} still pending"
        )
    return flushed


def _publish(producer: KafkaProducer, topic: str, event: dict):
    """
    Send one event to the given topic and block until acked (max 5s).

    On failure, the event is appended to a local buffer instead of being
    dropped silently — see flush_pending_events() for how it's retried.
    """
    flush_pending_events(producer)  # opportunistic: clear old backlog first

    future = producer.send(topic, value=event)
    try:
        future.get(timeout=5)
    except Exception as e:
        logger.error(f"Kafka publish failed (topic={topic}): {e} — buffering for retry")
        _append_to_buffer(topic, event)


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
