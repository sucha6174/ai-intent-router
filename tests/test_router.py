import os
import sys
import json
import pytest

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app import app, chat, ChatRequest
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


def test_chat_direct_calls():
    # Positional string invocation
    r1 = chat("Fix this python syntax error")
    assert r1["intent"] == "code"
    assert r1["confidence"] >= 0.7

    # Keyword argument invocation
    r2 = chat(message="Fix this python syntax error")
    assert r2["intent"] == "code"
    assert r2["confidence"] >= 0.7

    # Dictionary invocation
    r3 = chat({"message": "Fix this python syntax error"})
    assert r3["intent"] == "code"
    assert r3["confidence"] >= 0.7

    # Pydantic model invocation
    r4 = chat(ChatRequest(message="Fix this python syntax error"))
    assert r4["intent"] == "code"
    assert r4["confidence"] >= 0.7


def test_swagger_conflict_resolution():
    # Case A: Swagger default 'string' in query param, real user query in body
    res_a = client.post(
        "/chat?message=string",
        json={"message": "How do I optimize this python function?"}
    )
    assert res_a.status_code == 200
    assert res_a.json()["intent"] == "code"
    assert res_a.json()["confidence"] >= 0.7

    # Case B: Query param only with no body
    res_b = client.post(
        "/chat?message=How%20do%20I%20optimize%20this%20python%20function%3F"
    )
    assert res_b.status_code == 200
    assert res_b.json()["intent"] == "code"
    assert res_b.json()["confidence"] >= 0.7

    # Case C: JSON string in query param
    res_c = client.post('/chat?message={"message": "How do I optimize this python function?"}')
    assert res_c.status_code == 200
    assert res_c.json()["intent"] == "code"
    assert res_c.json()["confidence"] >= 0.7


def test_user_reported_intents():
    test_cases = [
        ("How do I optimize this python function?", "code", 0.7),
        ("Write a professional email asking for leave.", "writing", 0.7),
        ("Analyze this dataset and find the important trends.", "data", 0.7),
        ("How should I prepare for a software developer interview?", "career", 0.7),
        ("Hello", "unclear", 0.0),
    ]
    for prompt, expected_intent, min_conf in test_cases:
        res = client.post("/chat", json={"message": prompt})
        assert res.status_code == 200
        data = res.json()
        assert data["intent"] == expected_intent
        if expected_intent == "unclear":
            assert data["confidence"] < 0.7
        else:
            assert data["confidence"] >= min_conf

