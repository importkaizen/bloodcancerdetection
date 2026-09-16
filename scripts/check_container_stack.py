"""Check a running isolated Compose stack with synthetic records and task replay.

The caller starts the stack first. This retains synthetic records for inspection
and never starts, stops, or deletes services. Use a separate Compose project.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--project", default="hemasight-check")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--dashboard", default="http://127.0.0.1:3000")
    parser.add_argument("--fixture-models", action="store_true", help="Install synthetic model fixtures into a fresh disposable worker container; refuses existing models")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    compose = [args.docker, "compose", "-p", args.project, "-f", str(repo / "hemasight/docker/docker-compose.yml")]

    def request(url, payload=None, key=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Idempotency-Key"] = key
        body = json.dumps(payload).encode() if payload else None
        with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=20) as response:
            return response.status, response.read().decode()

    def sql(query):
        return subprocess.check_output(compose + ["exec", "-T", "postgres", "psql", "-U", "hemasight", "-d", "hemasight", "-tAc", query], text=True).strip()

    assert json.loads(request(args.api + "/health")[1])["status"] == "ok"
    assert json.loads(request(args.dashboard + "/api/health")[1])["status"] == "ok"
    if args.fixture_models:
        fixture_code = '''
from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
from hemasight.ml import model_training as training, anomaly
from hemasight.config import RISK_MODEL_PATH, SCALER_PATH, MODEL_CONFIG_PATH
paths = [RISK_MODEL_PATH, SCALER_PATH, MODEL_CONFIG_PATH, anomaly.ANOMALY_MODEL_PATH, anomaly.ANOMALY_CONFIG_PATH]
assert not any(path.exists() for path in paths), "Refusing to overwrite existing models"
matrix = np.random.default_rng(42).normal(size=(100, 13))
frame = pd.DataFrame(matrix, columns=training.FEATURE_COLUMNS)
frame["label"] = np.tile([0, 1], 50)
training.MODEL_VERSION = "synthetic_rf_v2"
anomaly.VERSION = "synthetic_if_v2"
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "synthetic.csv"
    frame.to_csv(path, index=False)
    training.train(str(path), model_type="rf", feature_version="v2")
anomaly.fit_isolation_forest(matrix, feature_version="v2")
for path in [MODEL_CONFIG_PATH, anomaly.ANOMALY_CONFIG_PATH]:
    config = json.loads(path.read_text())
    config["synthetic_fixture_only"] = True
    path.write_text(json.dumps(config))
print("Synthetic model fixtures installed in disposable worker container")
'''
        subprocess.run(compose + ["exec", "-T", "workers", "python", "-c", fixture_code], check=True)
    token = uuid.uuid4().hex
    payload = {"patient_id": "SYNTHETIC-CHECK-" + token, "date": "2024-01-01", "wbc": 7.2,
               "rbc": 4.8, "platelets": 210, "hemoglobin": 13.5, "lymphocytes": 40}
    status, body = request(args.api + "/blood-test", payload, token)
    assert status == 202
    blood_id = int(json.loads(body)["blood_test_id"])
    assert json.loads(request(args.api + "/blood-test", payload, token)[1])["blood_test_id"] == blood_id
    try:
        request(args.api + "/blood-test", {**payload, "wbc": 9}, token)
        raise AssertionError("Changed request reused the same idempotency key")
    except urllib.error.HTTPError as exc:
        assert exc.code == 409
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if sql(f"SELECT count(*) FROM features WHERE blood_test_id={blood_id}") == "1":
            break
        time.sleep(1)
    else:
        raise AssertionError("Queue-to-worker feature processing did not complete")
    assert sql(f"SELECT count(*) FROM ingestion_events WHERE blood_test_id={blood_id} AND published_at IS NOT NULL") == "1"
    assert sql(f"SELECT feature_version FROM features WHERE blood_test_id={blood_id}") == "v2"
    patient_id = int(sql(f"SELECT patient_id FROM blood_tests WHERE id={blood_id}"))
    assert '<div id="root">' in request(f"{args.dashboard}/patients/{patient_id}")[1]
    tests = json.loads(request(f"{args.dashboard}/api/patients/{patient_id}/blood-tests")[1])
    assert len(tests) == 1 and tests[0]["id"] == blood_id
    # Competing worker executions must reuse the same feature row under PG locks.
    code = (
        "from concurrent.futures import ThreadPoolExecutor; "
        "from hemasight.workers.feature_worker import process_blood_test; "
        f"pool=ThreadPoolExecutor(max_workers=4); results=list(pool.map(process_blood_test.run, [{blood_id}]*8)); "
        "assert len({r['feature_id'] for r in results}) == 1; pool.shutdown()"
    )
    subprocess.run(compose + ["exec", "-T", "workers", "python", "-c", code], check=True)
    assert sql(f"SELECT count(*) FROM features WHERE blood_test_id={blood_id}") == "1"
    if args.fixture_models:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            counts = sql(f"SELECT (SELECT count(*) FROM risk_scores WHERE blood_test_id={blood_id}), (SELECT count(*) FROM anomaly_scores WHERE blood_test_id={blood_id})")
            if counts == "1|1":
                break
            time.sleep(1)
        else:
            raise AssertionError("Synthetic model scoring did not complete exactly once")
        risk = json.loads(request(f"{args.dashboard}/api/patients/{patient_id}/risk-scores")[1])
        assert len(risk) == 1 and risk[0]["model_version"] == "synthetic_rf_v2"
        assert risk[0]["blood_test_date"].startswith("2024-01-01")
        assert 0 <= risk[0]["score"] <= 1
    print(json.dumps({"status": "passed", "patient_id": patient_id, "blood_test_id": blood_id,
                      "checks": ["API health", "Nginx proxy", "SPA refresh route", "HTTP idempotency",
                                 "conflicting retry", "durable delivery", "live worker processing",
                                 "PostgreSQL concurrent feature replay"] +
                                (["synthetic risk/anomaly scoring and replay"] if args.fixture_models else [])}, indent=2))


if __name__ == "__main__":
    main()
