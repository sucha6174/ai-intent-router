# AI Intent Router

## Overview

AI Intent Router is a FastAPI-based intelligent request routing system that uses a Large Language Model (LLM) to classify user intent and dynamically route requests to specialized AI personas.

The system demonstrates prompt engineering, intent classification, confidence-based routing, fallback handling, structured logging, and modular AI application design.

---

## Features

* LLM-powered intent classification
* Confidence score evaluation
* Dynamic routing to expert personas
* Fallback keyword-based classifier
* Structured interaction logging
* FastAPI REST API
* Modular architecture
* OpenAI API integration

---

## Supported Intents

| Intent  | Description                                     |
| ------- | ----------------------------------------------- |
| code    | Programming and software development assistance |
| data    | Data analysis and statistics guidance           |
| writing | Writing improvement and coaching                |
| career  | Career advice and interview preparation         |
| unclear | Requests requiring clarification                |

---

## Architecture

User Request
↓
Intent Classifier (LLM)
↓
Confidence Evaluation
↓
Intent Router
↓
Specialized Persona
↓
Response Generation
↓
Structured Logging

---

## Project Structure

```text
ai-intent-router/
├── app.py
├── classifier.py
├── router.py
├── prompts.py
├── logger.py
├── requirements.txt
└── README.md
```

## API Endpoint

### Chat Endpoint

```http
POST /chat
```

### Example Request

```json
{
  "message": "How do I optimize a SQL query?"
}
```

### Example Response

```json
{
  "intent": "code",
  "confidence": 0.92,
  "response": "You can optimize SQL queries by..."
}
```

---

## Routing Logic

1. User message is classified using GPT-4o Mini.
2. Intent and confidence score are returned.
3. Low-confidence requests trigger clarification.
4. Valid intents are routed to specialized AI personas.
5. Responses are logged for auditing and analysis.

---

## Fallback Strategy

If the OpenAI API is unavailable:

* Keyword-based classification is used.
* Specialist fallback responses are returned.
* The application remains operational.

---

## Logging

All routed requests are stored in:

```text
route_log.jsonl
```

Logged information:

* Intent
* Confidence Score
* User Message
* Generated Response

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
uvicorn app:app --reload
```

---

## Future Enhancements

* Docker support
* Automated testing
* Intent analytics dashboard
* Manual intent override
* Multi-agent routing

---

## Author

Kalari Srisucha
