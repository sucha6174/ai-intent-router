import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts import PROMPTS

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


def route_and_respond(message, intent_data):

    intent = intent_data["intent"]
    confidence = intent_data["confidence"]

    # Bonus: Confidence threshold
    if confidence < 0.7:
        return "I'm not sure I understood. Are you asking about coding, data analysis, writing, or career advice?"

    # Handle unclear intent
    if intent == "unclear":
        return "Could you clarify your request? Are you asking about coding, data analysis, writing improvement, or career advice?"

    # Get expert system prompt
    system_prompt = PROMPTS.get(intent)

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:

        print("Specialist Error:", e)

        # fallback response if API fails
        if intent == "code":
            return "Code expert selected. AI generation unavailable due to API quota."

        elif intent == "data":
            return "Data analyst selected. AI generation unavailable due to API quota."

        elif intent == "writing":
            return "Writing coach selected. AI generation unavailable due to API quota."

        elif intent == "career":
            return "Career advisor selected. AI generation unavailable due to API quota."

        else:
            return "AI response unavailable."