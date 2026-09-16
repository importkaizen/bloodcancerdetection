"""Retry committed ingestion events until the broker confirms delivery."""
import argparse
from datetime import datetime, timezone
import logging
import time

from sqlalchemy.orm import sessionmaker

from hemasight.data_pipeline.producer import publish_blood_test_ingested
from hemasight.db.models import BloodTest, IngestionEvent, Patient, get_engine

logger = logging.getLogger(__name__)


def dispatch_pending(*, session_factory=None, event_id=None, limit=100, publisher=None):
    """At-least-once publication; workers must tolerate delivery duplicates.

    PostgreSQL row locks serialize concurrent dispatchers. A crash after broker
    confirmation but before commit can redeliver the same blood-test ID.
    """
    session_factory = session_factory or sessionmaker(bind=get_engine())
    publisher = publisher or publish_blood_test_ingested
    delivered = 0
    for _ in range(limit):
        with session_factory() as db:
            with db.begin():
                query = db.query(IngestionEvent).filter(IngestionEvent.published_at.is_(None))
                if event_id is not None:
                    query = query.filter(IngestionEvent.id == event_id)
                event = query.order_by(IngestionEvent.id).with_for_update(skip_locked=True).first()
                if event is None:
                    break
                patient_id = (
                    db.query(Patient.external_id).join(BloodTest, BloodTest.patient_id == Patient.id)
                    .filter(BloodTest.id == event.blood_test_id).scalar()
                )
                publisher(event.blood_test_id, patient_id)
                event.published_at = datetime.now(timezone.utc)
            delivered += 1
    return delivered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    factory = sessionmaker(bind=get_engine())
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            count = dispatch_pending(session_factory=factory)
            if count:
                logger.info("Delivered %s ingestion events", count)
        except Exception:
            if args.once:
                raise
            logger.warning("Delivery unavailable; pending events retained for retry")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
