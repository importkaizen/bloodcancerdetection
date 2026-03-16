"""RabbitMQ producer for blood test ingestion."""
import json
import uuid
from typing import Any

import pika
from hemasight.config import BLOOD_TEST_QUEUE, RABBITMQ_URL


def publish_blood_test_ingested(blood_test_id: int, patient_id: str) -> str:
    """Publish a message to the blood_test.ingested queue. Returns delivery tag or message id."""
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=BLOOD_TEST_QUEUE, durable=True)
    body = json.dumps({"blood_test_id": blood_test_id, "patient_id": patient_id})
    msg_id = str(uuid.uuid4())
    channel.basic_publish(
        exchange="",
        routing_key=BLOOD_TEST_QUEUE,
        body=body,
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
            correlation_id=msg_id,
        ),
    )
    connection.close()
    return msg_id
