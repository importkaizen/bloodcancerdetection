"""RabbitMQ consumer: consume blood_test.ingested and enqueue Celery tasks."""
import json
import logging
import os
import sys
import time

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pika
from hemasight.config import BLOOD_TEST_QUEUE, RABBITMQ_URL
from hemasight.workers.feature_worker import process_blood_test


def on_message(channel, method, properties, body):
    try:
        payload = json.loads(body)
        blood_test_id = payload.get("blood_test_id") if isinstance(payload, dict) else None
        if type(blood_test_id) is not int or blood_test_id <= 0:
            raise ValueError("Invalid blood test ID")
    except (ValueError, UnicodeDecodeError):
        logging.getLogger(__name__).warning("Rejected malformed ingestion message")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return
    try:
        process_blood_test.delay(blood_test_id)
    except Exception:
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        logging.getLogger(__name__).warning("Task delivery unavailable; message requeued")
        return
    channel.basic_ack(delivery_tag=method.delivery_tag)


def run_consumer():
    params = pika.URLParameters(RABBITMQ_URL)
    params.socket_timeout = 5
    params.stack_timeout = 10
    params.blocked_connection_timeout = 10
    while True:
        connection = None
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=BLOOD_TEST_QUEUE, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=BLOOD_TEST_QUEUE, on_message_callback=on_message)
            channel.start_consuming()
            return
        except (pika.exceptions.AMQPError, OSError):
            logging.getLogger(__name__).warning("Queue connection unavailable; reconnecting in five seconds")
        finally:
            if connection is not None and connection.is_open:
                try:
                    connection.close()
                except pika.exceptions.AMQPError:
                    pass
        time.sleep(5)


if __name__ == "__main__":
    run_consumer()
