import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None

CLASSIFIER_PROMPT = """
Classify the user's intent.

Choose ONLY one:
code
data
writing
career
unclear

Return JSON like:
{"intent":"code","confidence":0.9}
"""


def classify_intent(message):

    try:

        if not client:
            raise ValueError("OpenAI API key not configured or client unavailable")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": message}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        print("LLM RESPONSE:", content)

        data = json.loads(content)

        return data

    except Exception as e:

        print("ERROR:", e)

        # fallback classifier
        msg = message.lower()

        # Code intent
        if "python" in msg or "code" in msg or "bug" in msg or "sql" in msg or "function" in msg:
            return {"intent": "code", "confidence": 0.8}

        # Data intent
        elif "average" in msg or "data" in msg or "pivot" in msg or "table" in msg or "numbers" in msg:
            return {"intent": "data", "confidence": 0.8}

        # Writing intent
        elif "write" in msg or "paragraph" in msg or "sentence" in msg or "writing" in msg or "verbose" in msg:
            return {"intent": "writing", "confidence": 0.8}

        # Career intent
        elif "job" in msg or "career" in msg or "interview" in msg or "resume" in msg:
            return {"intent": "career", "confidence": 0.8}

        else:
            return {"intent": "unclear", "confidence": 0.0}