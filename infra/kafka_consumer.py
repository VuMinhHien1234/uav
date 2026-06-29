"""Factory for KafkaConsumer — single place to configure broker."""
import json
from kafka import KafkaConsumer
from config import settings


def make_consumer(topic: str, group_id: str) -> KafkaConsumer:
    return KafkaConsumer(
        topic,
        bootstrap_servers=[settings.KAFKA_BROKER],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest", 
        group_id=group_id,
    )
