import os
import sys
import json
import pytest

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app import app
from classifier import classify_intent
from router import route_and_respond
from logger import log_file

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AI Intent Router"
    assert "chat" in data["endpoints"]


def test_chat_json_body_code_intent():
    response = client.post("/chat", json={"message": "How do I optimize this python function?"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "code"
    assert data["confidence"] >= 0.7
    assert "Code expert" in data["response"] or len(data["response"]) > 0


def test_chat_query_param():
    response = client.post("/chat?message=How do I optimize this python code?")
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "code"
    assert data["confidence"] >= 0.7


def test_chat_empty_message_error():
    response = client.post("/chat", json={})
    assert response.status_code == 400
    assert "required" in response.json()["detail"].lower()


def test_classifier_intents():
    # Code
    code_res = classify_intent("Fix the bug in this python function")
    assert code_res["intent"] == "code"
    assert code_res["confidence"] > 0

    # Data
    data_res = classify_intent("Calculate the average and show numbers in a pivot table")
    assert data_res["intent"] == "data"
    assert data_res["confidence"] > 0

    # Writing
    writing_res = classify_intent("Help me write a clear and concise paragraph")
    assert writing_res["intent"] == "writing"
    assert writing_res["confidence"] > 0

    # Career
    career_res = classify_intent("How can I prepare for a software engineer job interview?")
    assert career_res["intent"] == "career"
    assert career_res["confidence"] > 0

    # Unclear
    unclear_res = classify_intent("The clouds look nice today")
    assert unclear_res["intent"] == "unclear"
    assert unclear_res["confidence"] == 0.0


def test_router_confidence_threshold():
    # Low confidence (< 0.7) should return clarification
    low_conf_data = {"intent": "code", "confidence": 0.5}
    response = route_and_respond("test message", low_conf_data)
    assert "not sure i understood" in response.lower()


def test_router_unclear_intent():
    unclear_data = {"intent": "unclear", "confidence": 0.8}
    response = route_and_respond("random gibberish", unclear_data)
    assert "clarify your request" in response.lower()


def test_manual_intent_overrides():
    intents = [
        ("@code write a binary search algorithm", "code"),
        ("@data compute correlation matrix", "data"),
        ("@writing edit this introduction", "writing"),
        ("@career review my resume", "career"),
    ]
    for prompt, expected_intent in intents:
        response = client.post("/chat", json={"message": prompt})
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == expected_intent
        assert data["confidence"] == 1.0


def test_logging_functionality():
    test_msg = "test_logging_python_code_query"
    response = client.post("/chat", json={"message": test_msg})
    assert response.status_code == 200

    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        lines = f.readlines()
    assert len(lines) > 0
    last_entry = json.loads(lines[-1])
    assert "intent" in last_entry
    assert "confidence" in last_entry
    assert "user_message" in last_entry
    assert "final_response" in last_entry
    assert last_entry["user_message"] == test_msg
