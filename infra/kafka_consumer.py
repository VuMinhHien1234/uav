"""Factory for KafkaConsumer — single place to configure broker and topic."""
import json
from kafka import KafkaConsumer
from config import settings


def make_consumer(group_id: str) -> KafkaConsumer:
    return KafkaConsumer(
        settings.KAFKA_TOPIC,
        bootstrap_servers=[settings.KAFKA_BROKER],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        group_id=group_id,
    )
