from fastapi import FastAPI
from classifier import classify_intent
from router import route_and_respond
from logger import log_route

app = FastAPI()

@app.post("/chat")
def chat(message: str):

    # Manual intent override
    if message.startswith("@code "):
        intent_data = {"intent": "code", "confidence": 1.0}
        message = message.replace("@code ", "", 1)

    elif message.startswith("@data "):
        intent_data = {"intent": "data", "confidence": 1.0}
        message = message.replace("@data ", "", 1)

    elif message.startswith("@writing "):
        intent_data = {"intent": "writing", "confidence": 1.0}
        message = message.replace("@writing ", "", 1)

    elif message.startswith("@career "):
        intent_data = {"intent": "career", "confidence": 1.0}
        message = message.replace("@career ", "", 1)

    else:
        # Classify using LLM
        intent_data = classify_intent(message)

    # Route to specialized persona
    response_text = route_and_respond(message, intent_data)

    # Log interaction
    log_route(
        intent_data["intent"],
        intent_data["confidence"],
        message,
        response_text
    )

    return {
        "intent": intent_data["intent"],
        "confidence": intent_data["confidence"],
        "response": response_text
    }