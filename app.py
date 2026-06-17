from fastapi import FastAPI
from classifier import classify_intent
from router import route_and_respond
from logger import log_route

app = FastAPI()

@app.post("/chat")
def chat(message: str):
    # 1. Classify using LLM
    intent_data = classify_intent(message)
    
    # 2. Route to specialized persona
    response_text = route_and_respond(message, intent_data)

    # 3. Log the interaction
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