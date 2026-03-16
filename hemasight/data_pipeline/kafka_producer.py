"""Optional Kafka producer (use when BROKER=kafka). Requires: pip install confluent-kafka."""
from typing import Optional

from hemasight.config import BLOOD_TEST_QUEUE


def publish_blood_test_ingested_kafka(blood_test_id: int, patient_id: str, bootstrap_servers: str = "localhost:9092") -> Optional[str]:
    """Publish to Kafka topic (same logical queue as RabbitMQ). Returns message id or None if Kafka not used."""
    try:
        from confluent_kafka import Producer
        import json
        p = Producer({"bootstrap.servers": bootstrap_servers})
        payload = json.dumps({"blood_test_id": blood_test_id, "patient_id": patient_id})
        p.produce(BLOOD_TEST_QUEUE.replace(".", "_"), value=payload, key=patient_id.encode())
        p.flush()
        return str(blood_test_id)
    except Exception:
        return None
