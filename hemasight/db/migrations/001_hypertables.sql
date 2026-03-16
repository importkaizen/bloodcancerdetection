-- Enable TimescaleDB (run as superuser or in init)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Tables are created by SQLAlchemy Base.metadata.create_all.
-- This script converts them to hypertables after first deploy.
-- Run once after tables exist:

SELECT create_hypertable('blood_tests', 'created_at', if_not_exists => TRUE);
SELECT create_hypertable('features', 'computed_at', if_not_exists => TRUE);
SELECT create_hypertable('risk_scores', 'computed_at', if_not_exists => TRUE);
