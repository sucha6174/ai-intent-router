from typing import Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from classifier import classify_intent
from router import route_and_respond
from logger import log_route

app = FastAPI(
    title="AI Intent Router",
    description="Intelligent request routing system using LLM intent classification and specialized AI personas",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    message: Optional[str] = None


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
def chat(payload: Optional[Union[ChatRequest, dict, str]] = None, message: Optional[str] = None):
    # Support direct string argument, dict, ChatRequest model, or query parameter
    user_message = None
    if isinstance(payload, str):
        user_message = payload
    elif isinstance(payload, dict):
        user_message = payload.get("message")
    elif isinstance(payload, ChatRequest):
        user_message = payload.message
    elif hasattr(payload, "message"):
        user_message = payload.message

    if not user_message and message:
        user_message = message

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