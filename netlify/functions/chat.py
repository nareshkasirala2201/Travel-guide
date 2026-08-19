import json
import os

from openai import AzureOpenAI


SYSTEM_PROMPT = (
    "You are a thoughtful trip-planning expert. Help travelers design realistic, "
    "enjoyable itineraries tailored to their dates, budget, interests, mobility, "
    "and travel style. Ask focused follow-up questions when important details are "
    "missing. Give practical advice about timing, transport, neighborhoods, costs, "
    "and alternatives. Be clear about uncertainty and remind users to verify live "
    "prices, schedules, entry rules, weather, and other time-sensitive details."
)


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-store"},
        "body": json.dumps(body),
    }


def handler(event, context):
    if event.get("httpMethod") != "POST":
        return response(405, {"detail": "Method not allowed."})

    endpoint = os.getenv("AZURE_ENDPOINT")
    deployment = os.getenv("AZURE_DEPLOYMENT")
    api_key = os.getenv("AZURE_API_KEY")
    if not endpoint or not deployment or not api_key:
        return response(500, {"detail": "The trip planner is not configured in Netlify."})

    try:
        payload = json.loads(event.get("body") or "{}")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or len(messages) > 100:
            return response(400, {"detail": "A non-empty messages list is required."})
        clean_messages = []
        for message in messages:
            if message.get("role") not in ("user", "assistant"):
                return response(400, {"detail": "Messages may only be user or assistant messages."})
            content = message.get("content")
            if not isinstance(content, str) or not content.strip() or len(content) > 12000:
                return response(400, {"detail": "Each message must have valid content."})
            clean_messages.append({"role": message["role"], "content": content})

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
        )
        completion = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *clean_messages],
        )
        reply = completion.choices[0].message.content
        if not reply:
            return response(502, {"detail": "The AI service returned an empty reply."})
        return response(200, {"reply": reply})
    except json.JSONDecodeError:
        return response(400, {"detail": "The request body must be valid JSON."})
    except Exception as error:
        print(f"Azure chat request failed: {type(error).__name__}")
        return response(502, {"detail": "The trip planner could not reach the AI service. Please try again."})