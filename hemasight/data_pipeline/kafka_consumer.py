"""Optional Kafka consumer (use when BROKER=kafka). Requires: pip install confluent-kafka."""
from hemasight.config import BLOOD_TEST_QUEUE


def run_kafka_consumer(bootstrap_servers: str = "localhost:9092"):
    """Consume from Kafka topic and enqueue Celery process_blood_test tasks."""
    from confluent_kafka import Consumer
    import json
    from hemasight.workers.feature_worker import process_blood_test
    c = Consumer({"bootstrap.servers": bootstrap_servers, "group.id": "hemasight-consumer"})
    topic = BLOOD_TEST_QUEUE.replace(".", "_")
    c.subscribe([topic])
    while True:
        msg = c.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            continue
        try:
            payload = json.loads(msg.value().decode())
            blood_test_id = payload.get("blood_test_id")
            if blood_test_id is not None:
                process_blood_test.delay(blood_test_id)
        except Exception:
            pass
