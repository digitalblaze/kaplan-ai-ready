"""
Kaplan AI Ready — FastAPI Backend
Proxies AI requests from the frontend to Gemini, keeping the API key server-side.
"""

import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Kaplan AI Ready API", version="1.0.0")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://kaplan-ai-ready.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


class Message(BaseModel):
    role: str
    content: str


class ClaudeRequest(BaseModel):
    messages: list[Message] = []
    system: str | None = None
    max_tokens: int = 1000
    model: str = GEMINI_MODEL  # ignored — we always use GEMINI_MODEL


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "kaplan-ai-ready"}


@app.post("/api/claude")
async def claude_proxy(request: ClaudeRequest):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    # Anthropic role "assistant" → Gemini role "model"
    contents = [
        {
            "role": "model" if m.role == "assistant" else "user",
            "parts": [{"text": m.content}],
        }
        for m in request.messages
    ]

    # Simulator requests have no system prompt and expect structured JSON back.
    # Use Gemini's JSON mode to guarantee a complete, valid JSON response.
    is_json_request = request.system is None

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max(request.max_tokens, 4096),
            **({"responseMimeType": "application/json"} if is_json_request else {}),
        },
    }

    if request.system:
        body["systemInstruction"] = {"parts": [{"text": request.system}]}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GEMINI_URL,
            params={"key": key},
            json=body,
        )

    data = resp.json()

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=data.get("error", {}).get("message", f"Gemini API {resp.status_code}"),
        )

    # Return in Anthropic response shape so frontend error handling works unchanged
    text = "".join(
        part.get("text", "")
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    )
    return JSONResponse({"content": [{"type": "text", "text": text}]})
