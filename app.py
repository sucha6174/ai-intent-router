import json
from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from classifier import classify_intent
from router import route_and_respond
from logger import log_route

app = FastAPI(
    title="AI Intent Router",
    description="Intelligent request routing system using LLM intent classification and specialized AI personas",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    message: Optional[str] = Field(
        default=None,
        description="User query to classify and route",
        json_schema_extra={"example": "How do I optimize this python function?"}
    )


def extract_user_message(payload, message):
    # 1. Extract from payload (JSON body, dict, object, or string)
    if payload is not None:
        if isinstance(payload, str):
            trimmed = payload.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                    if isinstance(data, dict) and "message" in data and data["message"]:
                        return str(data["message"]).strip()
                except Exception:
                    pass
            if trimmed and trimmed != "string":
                return trimmed
        elif isinstance(payload, dict):
            if "message" in payload and payload["message"] and str(payload["message"]).strip() != "string":
                return str(payload["message"]).strip()
            elif "message" in payload and payload["message"]:
                return str(payload["message"]).strip()
        elif hasattr(payload, "message") and payload.message:
            val = str(payload.message).strip()
            if val != "string":
                return val

    # 2. Extract from query parameter
    if message is not None and str(message).strip() != "" and str(message).strip() != "string":
        trimmed = str(message).strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                data = json.loads(trimmed)
                if isinstance(data, dict) and "message" in data and data["message"]:
                    return str(data["message"]).strip()
            except Exception:
                pass
        return trimmed

    # Fallback to payload or query if only "string" was provided
    if payload is not None:
        if isinstance(payload, str) and payload.strip():
            return payload.strip()
        elif hasattr(payload, "message") and payload.message:
            return str(payload.message).strip()
        elif isinstance(payload, dict) and "message" in payload and payload["message"]:
            return str(payload["message"]).strip()
    if message is not None and str(message).strip():
        return str(message).strip()

    return None


@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "AI Intent Router",
        "endpoints": {
            "chat": "/chat (POST)",
            "docs": "/docs (GET)"
        }
    }


@app.post("/chat")
def chat(
    payload: Optional[ChatRequest] = None,
    message: Optional[str] = Query(default=None, include_in_schema=False)
):
    # Support direct string argument, dict, ChatRequest model, or query parameter
    user_message = extract_user_message(payload, message)

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="A non-empty 'message' field in the body or query parameter is required."
        )

    # Manual intent override
    if user_message.startswith("@code "):
        intent_data = {"intent": "code", "confidence": 1.0}
        user_message = user_message.replace("@code ", "", 1)

    elif user_message.startswith("@data "):
        intent_data = {"intent": "data", "confidence": 1.0}
        user_message = user_message.replace("@data ", "", 1)

    elif user_message.startswith("@writing "):
        intent_data = {"intent": "writing", "confidence": 1.0}
        user_message = user_message.replace("@writing ", "", 1)

    elif user_message.startswith("@career "):
        intent_data = {"intent": "career", "confidence": 1.0}
        user_message = user_message.replace("@career ", "", 1)

    else:
        # Classify using LLM
        intent_data = classify_intent(user_message)

    # Route to specialized persona
    response_text = route_and_respond(user_message, intent_data)

    # Log interaction
    log_route(
        intent_data["intent"],
        intent_data["confidence"],
        user_message,
        response_text
    )

    return {
        "intent": intent_data["intent"],
        "confidence": intent_data["confidence"],
        "response": response_text
    }