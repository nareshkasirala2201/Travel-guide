import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import AzureOpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Trip Planning Expert")

SYSTEM_PROMPT = (
    "You are a thoughtful trip-planning expert. Help travelers design realistic, "
    "enjoyable itineraries tailored to their dates, budget, interests, mobility, "
    "and travel style. Ask focused follow-up questions when important details are "
    "missing. Give practical advice about timing, transport, neighborhoods, costs, "
    "and alternatives. Be clear about uncertainty and remind users to verify live "
    "prices, schedules, entry rules, weather, and other time-sensitive details."
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


class ChatResponse(BaseModel):
    reply: str


def get_azure_client() -> tuple[AzureOpenAI, str]:
    endpoint = os.getenv("AZURE_ENDPOINT")
    deployment = os.getenv("AZURE_DEPLOYMENT")
    api_key = os.getenv("AZURE_API_KEY")

    if not endpoint or not deployment or not api_key:
        raise HTTPException(
            status_code=500,
            detail="The trip planner is not configured. Set the Azure environment variables and try again.",
        )

    return (
        AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
        ),
        deployment,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    client, deployment = get_azure_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(message.model_dump() for message in request.messages)

    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=messages,
        )
    except Exception as error:
        # Keep provider details and credentials out of the browser response.
        print(f"Azure chat request failed: {type(error).__name__}")
        raise HTTPException(
            status_code=502,
            detail="The trip planner could not reach the AI service. Please try again.",
        ) from error

    reply = completion.choices[0].message.content
    if not reply:
        raise HTTPException(status_code=502, detail="The AI service returned an empty reply.")
    return ChatResponse(reply=reply)