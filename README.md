# AI Intent Router

An intelligent, resilient request-routing service built with **FastAPI** and **OpenAI**. The system dynamically classifies user intent using LLMs, evaluates confidence scores, routes prompts to domain-expert personas, and provides robust keyword-based and response fallbacks if external AI services are unreachable or unconfigured.

---

## Overview

Modern AI applications often suffer when a single generic system prompt is forced to handle diverse domains such as software engineering, data science, editorial writing, and career consulting. 

**AI Intent Router** addresses this by acting as an intelligent orchestration gateway:
1. **Analyzes and Classifies:** Inspects incoming user queries using an LLM classifier (`gpt-4o-mini`) to determine domain intent and associated confidence.
2. **Evaluates Confidence:** Enforces a minimum confidence threshold (0.70). Ambiguous or low-confidence queries automatically trigger targeted clarification prompts rather than hallucinated responses.
3. **Routes to Domain Experts:** Selects specialized personas tailored specifically for coding, data analytics, writing, or career guidance.
4. **Resilient Two-Tier Fallback:** If the OpenAI API key is unset, rate-limited, or network unreachable, the router seamlessly drops back to a keyword-matching heuristic classifier and specialist quota fallbacks.
5. **Audits Every Query:** Automatically serializes all interactions (intent, confidence score, user message, and final response) into structured JSON Lines (`route_log.jsonl`).

---

## Features

- **LLM-Powered Intent Classification:** Analyzes prompts using OpenAI's `gpt-4o-mini` with strict JSON schema outputs.
- **Confidence Scoring & Thresholding:** Evaluates classification confidence; requests below `0.70` prompt the user for clarification.
- **Specialized Personas:** Tailored prompts for four distinct domains:
  - **Code Expert:** Senior software engineer offering clean code, logic explanations, and best practices.
  - **Data Analyst:** Quantitative analyst explaining statistics, patterns, and visualization recommendations.
  - **Writing Coach:** Editorial mentor focusing on tone, clarity, and grammatical structure without rewriting.
  - **Career Advisor:** Actionable career strategist asking targeted questions and recommending concrete next steps.
- **Manual Intent Overrides:** Power users can bypass LLM classification using `@code`, `@data`, `@writing`, or `@career` command prefixes (sets confidence to 1.0).
- **Graceful Fallback Mechanism:** Includes rule-based keyword matching and static specialist responses when API credentials are unconfigured or external services are down.
- **Dual Request Ingestion:** Supports standard REST JSON bodies (`{"message": "..."}`) as well as URL query parameters (`?message=...`).
- **Structured Audit Logging:** Appends every routing transaction to a path-portable `route_log.jsonl` file.
- **Automated Test Suite:** Comprehensive `pytest` test suite verifying endpoints, classification, routing thresholds, overrides, and logging.

---

## Architecture & Workflow

```
User Request (JSON Body / Query Param)
               │
               ▼
     Manual Override Prefix?
     ├── YES ──► Force Intent (@code, @data, @writing, @career) & Confidence 1.0
     └── NO  ──► Classify Intent via LLM (gpt-4o-mini)
                   │ (Network / Key Error)
                   └──► Keyword Fallback Classifier
               │
               ▼
    Confidence Evaluation
     ├── Confidence < 0.70 or "unclear" ──► Return Clarification Question
     └── Confidence >= 0.70             ──► Route to Specialist Persona
                                               │
                                               ▼
                                      Execute Specialist LLM
                                        │ (Network / Key Error)
                                        └──► Static Quota Fallback Response
                                               │
                                               ▼
                                      Log to route_log.jsonl
                                               │
                                               ▼
                                    Return JSON Response
```

---

## Supported Intents

| Intent | Persona | Core Objective | Fallback Keywords |
|---|---|---|---|
| `code` | Senior Software Engineer | Clean code, debugging, architecture, best practices | `python`, `code`, `bug`, `sql`, `function` |
| `data` | Professional Data Analyst | Statistical guidance, distributions, charts, anomaly detection | `average`, `data`, `pivot`, `table`, `numbers` |
| `writing` | Editorial Writing Coach | Structure, clarity, tone improvement (coaching mode) | `write`, `paragraph`, `sentence`, `writing`, `verbose` |
| `career` | Actionable Career Advisor | Interview prep, resume strategy, concrete next steps | `job`, `career`, `interview`, `resume` |
| `unclear` | System Clarifier | Prompts user to specify domain | Fallback default when no keywords match |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI | High-performance asynchronous REST API |
| **ASGI Server** | Uvicorn | Lightweight production ASGI web server |
| **Data Validation** | Pydantic v2 | Request/response schema validation |
| **AI / LLM** | OpenAI API (`gpt-4o-mini`) | Classification and expert persona response generation |
| **Configuration** | python-dotenv | Secure environment variable management |
| **Testing** | Pytest & HTTPX / TestClient | Automated unit and integration testing |
| **Logging** | Python JSON / File I/O | Structured append-only audit trail (`route_log.jsonl`) |

---

## Project Structure

```text
ai-intent-router/
├── app.py              # FastAPI application definition and endpoint routing
├── classifier.py       # LLM intent classification and keyword fallback logic
├── router.py           # Persona routing, confidence thresholding, and response generation
├── prompts.py          # Domain-expert system prompts
├── logger.py           # Structured JSONL logging utility
├── tests/
│   └── test_router.py  # Pytest test suite covering all endpoints and logic
├── .env.example        # Template for required environment variables
├── .gitignore          # Git exclusion rules (.env, logs, pycache, venv)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## How It Works

1. **`app.py`**: Declares FastAPI routes (`GET /`, `POST /chat`). Extracts the user query from either a JSON body (`ChatRequest`) or query parameter. Checks for manual command prefixes (`@code `, `@data `, `@writing `, `@career `) to bypass classification, calls the router, logs the interaction, and returns the response.
2. **`classifier.py`**: Sends the user query with `CLASSIFIER_PROMPT` to OpenAI with temperature 0. If the API key is not present or an exception occurs, it falls back to keyword matching to ensure zero downtime.
3. **`router.py`**: Evaluates whether confidence meets the 0.70 threshold. If intent is `unclear` or below threshold, asks the user for clarification. Otherwise, injects the query into the matching specialist prompt from `prompts.py` and returns the generated advice.
4. **`prompts.py`**: Contains carefully engineered system prompts defining the behavior, boundaries, and tone of each expert persona.
5. **`logger.py`**: Serializes each request, detected intent, confidence score, and generated response into `route_log.jsonl`.

---

## Installation

### Prerequisites

- Python 3.10+ installed
- Git installed
- OpenAI API Key (optional for testing; fallback works without key)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sucha6174/ai-intent-router.git
   cd ai-intent-router
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Environment Variables

Create a `.env` file in the root directory by copying the provided example:

```bash
cp .env.example .env
```

Edit `.env` and provide your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

> **Note:** If `OPENAI_API_KEY` is not provided, the application will automatically run in **fallback mode**, allowing full local testing without incurring API costs.

---

## Running the Application

Start the development server with Uvicorn:

```bash
uvicorn app:app --reload
```

The server will start at:
- **API Base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **Alternative ReDoc:** `http://127.0.0.1:8000/redoc`

---

## API Usage

### 1. Health & Service Info

- **Endpoint:** `GET /`
- **Description:** Returns API status and available endpoints.

#### Example Request:
```bash
curl -X GET http://127.0.0.1:8000/
```

#### Example Response:
```json
{
  "status": "healthy",
  "service": "AI Intent Router",
  "endpoints": {
    "chat": "/chat (POST)",
    "docs": "/docs (GET)"
  }
}
```

---

### 2. Chat & Route

- **Endpoint:** `POST /chat`
- **Description:** Classifies user query, routes to domain expert, and generates response.
- **Interactive Swagger UI (`/docs`):** Provides a clean, direct **Message** text input box — type any query and click **Execute** (no JSON editing required).
- **HTTP Methods Supported:** URL query parameter (`?message=...`) or JSON request body (`{"message": "..."}`).

#### Example 1: Direct Message Query (Swagger UI & URL parameter)
```bash
curl -X POST "http://127.0.0.1:8000/chat?message=How%20should%20I%20prepare%20for%20a%20software%20developer%20interview%3F"
```

#### Example 2: REST JSON Body
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How should I prepare for a software developer interview?"}'
```

#### Response Format:
```json
{
  "intent": "career",
  "confidence": 0.8,
  "response": "Career advisor selected. AI generation unavailable due to API quota."
}
```

---

### Example Scenarios

#### Scenario A: Automated Coding Route
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How do I fix a recursion depth error in Python?"}'
```

#### Scenario B: Manual Intent Override (`@writing`)
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "@writing Can you give me feedback on the opening paragraph of my cover letter?"}'
```

#### Scenario C: Low Confidence / Clarification Request
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What do you think about tomorrow?"}'
```

**Response:**
```json
{
  "intent": "unclear",
  "confidence": 0.0,
  "response": "I'm not sure I understood. Are you asking about coding, data analysis, writing, or career advice?"
}
```

---

## Testing

The project includes an automated test suite using `pytest`.

To run all tests:

```bash
pytest -v
```

### Test Coverage

The test suite in `tests/test_router.py` verifies:
- `test_root_endpoint`: Verifies API root health endpoint returns 200 OK and service metadata.
- `test_chat_json_body_code_intent`: Verifies JSON body ingestion, intent classification, and specialist response.
- `test_chat_query_param`: Verifies backward-compatible URL query parameter support (`/chat?message=...`).
- `test_chat_empty_message_error`: Verifies 400 Bad Request when an empty payload is submitted.
- `test_classifier_intents`: Unit tests individual intents (`code`, `data`, `writing`, `career`, `unclear`).
- `test_router_confidence_threshold`: Verifies requests below 0.70 confidence trigger clarification.
- `test_router_unclear_intent`: Verifies `unclear` intent handling.
- `test_manual_intent_overrides`: Verifies `@code`, `@data`, `@writing`, and `@career` prefixes bypass classification with 1.0 confidence.
- `test_logging_functionality`: Verifies interactions are appended correctly to `route_log.jsonl`.
- `test_chat_direct_calls`: Verifies programmatic Python function calls (positional string, keyword, dict, and Pydantic model).
- `test_swagger_conflict_resolution`: Verifies robust resolution of Swagger UI default parameters and JSON payload extraction.
- `test_user_reported_intents`: Verifies all 5 user query scenarios (coding, data, writing, career, unclear) via JSON body.
- `test_swagger_message_input_all_5_intents`: Verifies all 5 user query scenarios via Swagger UI message input (query parameter).

---

## Error Handling & Fallback

The application is architected with defensive error handling:

1. **Client Initialization Safety:** OpenAI client initialization is guarded so that missing environment variables do not cause runtime crashes on startup.
2. **Keyword Heuristic Fallback:** If the OpenAI API throws an exception (e.g., rate limit, invalid key, timeout), `classifier.py` automatically falls back to regex/keyword rules to classify the message.
3. **Specialist Quota Fallback:** If the generation step fails, `router.py` returns clear, domain-specific fallback responses indicating service state rather than throwing an unhandled HTTP 500.
4. **Input Validation:** Empty or malformed inputs return standard HTTP 400 with actionable error messages.

---

## Logging

All routed interactions are logged to `route_log.jsonl` using JSON Lines format.

Example log entry:

```json
{
  "intent": "code",
  "confidence": 0.95,
  "user_message": "How do I optimize a slow SQL join in PostgreSQL?",
  "final_response": "To optimize a slow SQL join in PostgreSQL, start by running EXPLAIN ANALYZE..."
}
```

The log path is resolved relative to `logger.py`, ensuring portability across operating systems and execution directories.

---

## Future Improvements

- **Semantic Vector Routing:** Embedding-based intent matching via cosine similarity for domain classification.
- **Streaming Response Support:** Server-Sent Events (SSE) for real-time persona streaming tokens.
- **Multi-Turn Context:** Session memory to track conversation threads per user.
- **Analytics Dashboard:** Web UI displaying intent distributions, average confidence scores, and request frequency.

---

## Author

**KALARI SRISUCHA**  
GitHub: [https://github.com/sucha6174](https://github.com/sucha6174)  
Repository: [https://github.com/sucha6174/ai-intent-router](https://github.com/sucha6174/ai-intent-router)

---

## Demo

Video Demo: To be added
