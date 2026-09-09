import json
import anyio
from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Query, Request
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


def extract_user_message(message=None, request: Request = None, payload=None):
    candidates = []

    # 1. Direct message argument (string, dict, ChatRequest, or query parameter)
    if message is not None:
        if isinstance(message, str):
            trimmed = message.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                    if isinstance(data, dict) and "message" in data and data["message"]:
                        candidates.append(str(data["message"]).strip())
                except Exception:
                    pass
            if trimmed:
                candidates.append(trimmed)
        elif isinstance(message, dict):
            if "message" in message and message["message"]:
                candidates.append(str(message["message"]).strip())
        elif hasattr(message, "message") and message.message:
            candidates.append(str(message.message).strip())

    # 2. Payload argument if provided directly in Python invocations
    if payload is not None:
        if isinstance(payload, str):
            trimmed = payload.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                    if isinstance(data, dict) and "message" in data and data["message"]:
                        candidates.append(str(data["message"]).strip())
                except Exception:
                    pass
            if trimmed:
                candidates.append(trimmed)
        elif isinstance(payload, dict):
            if "message" in payload and payload["message"]:
                candidates.append(str(payload["message"]).strip())
        elif hasattr(payload, "message") and payload.message:
            candidates.append(str(payload.message).strip())

    # 3. HTTP Request body (if request object is provided)
    if request is not None and isinstance(request, Request):
        try:
            body_bytes = anyio.from_thread.run(request.body)
            if body_bytes:
                body_str = body_bytes.decode("utf-8", errors="ignore").strip()
                if body_str:
                    try:
                        data = json.loads(body_str)
                        if isinstance(data, dict) and "message" in data and data["message"]:
                            candidates.append(str(data["message"]).strip())
                        elif isinstance(data, str) and data.strip():
                            candidates.append(data.strip())
                    except Exception:
                        candidates.append(body_str)
        except Exception:
            pass

    # Strip empty and prioritize non-placeholder strings
    valid_candidates = [str(c).strip() for c in candidates if c is not None and str(c).strip() != ""]
    non_placeholder = [c for c in valid_candidates if c != "string"]
    if non_placeholder:
        return non_placeholder[0]
    if valid_candidates:
        return valid_candidates[0]

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
    message: Optional[str] = Query(
        default=None,
        description="Type your message here..."
    ),
    request: Request = None
):
    # Support query parameter, direct string argument, dict, ChatRequest model, or JSON body
    user_message = extract_user_message(message=message, request=request)

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