"""Stop/restart RabbitMQ in a disposable hemasight-check project to verify recovery.

Uses synthetic records only; restores the broker in a finally block. This is an
explicit fault test and must not be run against a deployment serving users.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--project", default="hemasight-check")
    args = parser.parse_args()
    if not args.project.startswith("hemasight-check"):
        parser.error("Use an isolated project whose name starts with hemasight-check")
    repo = Path(__file__).resolve().parents[1]
    compose = [args.docker, "compose", "-p", args.project, "-f", str(repo / "hemasight/docker/docker-compose.yml")]

    def sql(query):
        return subprocess.check_output(compose + ["exec", "-T", "postgres", "psql", "-U", "hemasight", "-d", "hemasight", "-tAc", query], text=True).strip()

    token = uuid.uuid4().hex
    payload = {"patient_id": "SYNTHETIC-OUTAGE-" + token, "date": "2024-01-02", "wbc": 8,
               "rbc": 4.5, "platelets": 200, "hemoglobin": 13, "lymphocytes": 35}
    try:
        subprocess.run(compose + ["stop", "--timeout", "10", "rabbitmq"], check=True)
        request = urllib.request.Request("http://127.0.0.1:8000/blood-test", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", "Idempotency-Key": token})
        with urllib.request.urlopen(request, timeout=25) as response:
            assert response.status == 202
            blood_id = int(json.load(response)["blood_test_id"])
        assert sql(f"SELECT count(*) FROM ingestion_events WHERE blood_test_id={blood_id} AND published_at IS NULL") == "1"
        assert sql(f"SELECT count(*) FROM features WHERE blood_test_id={blood_id}") == "0"
        print("PASS: outage request saved as a pending event", flush=True)
    finally:
        subprocess.run(compose + ["up", "--no-deps", "-d", "--wait", "--wait-timeout", "90", "rabbitmq"], check=True)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if sql(f"SELECT count(*) FROM features WHERE blood_test_id={blood_id}") == "1":
            break
        time.sleep(1)
    else:
        raise AssertionError("Pending event did not reach a worker after broker recovery")
    assert sql(f"SELECT count(*) FROM ingestion_events WHERE blood_test_id={blood_id} AND published_at IS NOT NULL") == "1"
    print(json.dumps({"status": "passed", "blood_test_id": blood_id,
                      "checks": ["202 during broker outage", "pending delivery retained", "automatic processing after recovery"]}, indent=2))


if __name__ == "__main__":
    main()
