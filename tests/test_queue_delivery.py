"""Broker-boundary failure behavior without a running RabbitMQ server."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hemasight.data_pipeline import consumer, producer


@pytest.mark.parametrize("body", [b"broken", b"[]", b"{}", b'{"blood_test_id":true}', b'{"blood_test_id":-1}'])
def test_malformed_messages_do_not_poison_queue(monkeypatch, body):
    task = Mock()
    monkeypatch.setattr(consumer.process_blood_test, "delay", task)
    channel = Mock()
    consumer.on_message(channel, SimpleNamespace(delivery_tag=4), None, body)
    channel.basic_nack.assert_called_once_with(delivery_tag=4, requeue=False)
    channel.basic_ack.assert_not_called()
    task.assert_not_called()


def test_ack_only_after_task_accepted_and_requeue_transient_failure(monkeypatch):
    task = Mock(side_effect=RuntimeError("temporary"))
    monkeypatch.setattr(consumer.process_blood_test, "delay", task)
    channel = Mock()
    consumer.on_message(channel, SimpleNamespace(delivery_tag=5), None, b'{"blood_test_id":1}')
    channel.basic_nack.assert_called_once_with(delivery_tag=5, requeue=True)
    channel.basic_ack.assert_not_called()
    task.side_effect = None
    consumer.on_message(channel, SimpleNamespace(delivery_tag=6), None, b'{"blood_test_id":1}')
    channel.basic_ack.assert_called_once_with(delivery_tag=6)


@pytest.mark.parametrize("fails", [False, True])
def test_publisher_requires_confirmation_and_closes_connection(monkeypatch, fails):
    connection = Mock(is_open=True)
    channel = connection.channel.return_value
    if fails:
        channel.basic_publish.side_effect = RuntimeError("broker rejected publish")
    monkeypatch.setattr(producer.pika, "BlockingConnection", Mock(return_value=connection))
    if fails:
        with pytest.raises(RuntimeError):
            producer.publish_blood_test_ingested(1, "fixture")
    else:
        assert producer.publish_blood_test_ingested(1, "fixture")
    channel.confirm_delivery.assert_called_once()
    assert channel.basic_publish.call_args.kwargs["mandatory"] is True
    assert channel.basic_publish.call_args.kwargs["properties"].delivery_mode == 2
    connection.close.assert_called_once()


def test_consumer_reconnects_after_startup_and_stream_failures(monkeypatch):
    broken = Mock(is_open=False)
    broken.channel.return_value.start_consuming.side_effect = consumer.pika.exceptions.StreamLostError("disconnected")
    recovered = Mock(is_open=True)
    connect = Mock(side_effect=[consumer.pika.exceptions.AMQPConnectionError(), broken, recovered])
    monkeypatch.setattr(consumer.pika, "BlockingConnection", connect)
    pause = Mock()
    monkeypatch.setattr(consumer.time, "sleep", pause)
    consumer.run_consumer()
    assert connect.call_count == 3
    assert pause.call_count == 2
    recovered.channel.return_value.basic_consume.assert_called_once()
    recovered.channel.return_value.start_consuming.assert_called_once()
    recovered.close.assert_called_once()
