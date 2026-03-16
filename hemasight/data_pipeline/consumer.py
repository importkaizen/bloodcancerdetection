"""RabbitMQ consumer: consume blood_test.ingested and enqueue Celery tasks."""
import json
import os
import sys

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pika
from hemasight.config import BLOOD_TEST_QUEUE, RABBITMQ_URL
from hemasight.workers.feature_worker import process_blood_test


def on_message(channel, method, properties, body):
    try:
        payload = json.loads(body)
        blood_test_id = payload.get("blood_test_id")
        if blood_test_id is not None:
            process_blood_test.delay(blood_test_id)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        raise e


def run_consumer():
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=BLOOD_TEST_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=BLOOD_TEST_QUEUE, on_message_callback=on_message)
    channel.start_consuming()


if __name__ == "__main__":
    run_consumer()
