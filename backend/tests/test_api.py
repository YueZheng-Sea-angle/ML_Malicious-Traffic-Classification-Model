"""后端接口冒烟测试。

运行：cd project/backend && pytest -v
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.storage import get_store


@pytest.fixture()
def client():
    get_store().clear()
    with TestClient(app) as c:
        yield c


def _upload(client, name: str = "demo.pcap"):
    return client.post(
        "/api/traffic/upload",
        files={"file": (name, b"\xd4\xc3\xb2\xa1demo-traffic-bytes" * 64, "application/octet-stream")},
    )


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["inference_mode"] in {"model", "heuristic", "unavailable"}


def test_upload_rejects_bad_suffix(client):
    response = _upload(client, "readme.txt")
    assert response.status_code == 400


def test_upload_and_classify(client):
    upload = _upload(client)
    assert upload.status_code == 200
    file_id = upload.json()["file_id"]

    created = client.post("/api/tasks", json={"file_id": file_id, "max_flows": 8})
    assert created.status_code == 202
    task_id = created.json()["task_id"]

    task = _wait_for_completion(client, task_id)
    assert task["status"] == "succeeded", task.get("error")

    result = task["result"]
    assert result["flow_count"] >= 1
    assert 0.0 <= result["confidence"] <= 1.0
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-3) == 1.0
    assert len(result["flows"]) == result["flow_count"]


def test_task_not_found(client):
    assert client.get("/api/tasks/not-exist").status_code == 404


def test_create_task_with_unknown_file(client):
    assert client.post("/api/tasks", json={"file_id": "nope"}).status_code == 404


def test_models_and_stats(client):
    models = client.get("/api/models")
    assert models.status_code == 200
    assert models.json()["total"] >= 1

    stats = client.get("/api/tasks/stats")
    assert stats.status_code == 200
    assert "total_tasks" in stats.json()


def _wait_for_completion(client, task_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/tasks/{task_id}").json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        time.sleep(0.2)
    raise AssertionError("任务超时未完成")
