JARVIS Backend 

Python FastAPI backend for the JARVIS Android app. It reuses the existing OpenAI API integration and keeps the OpenAI key on the server.


Features


GET / — service status

GET /health and GET /api/health — health check (Android-compatible)

POST /chat — backward-compatible chat route

POST /api/chat — Android-compatible chat route (reply, session_id, intent)

POST /api/call/event — acknowledges an Android call lifecycle event; it does not answer calls or access cellular audio

POST /api/call/transcript — generates a summary from transcript text actually supplied by the app


Environment variables


OPENAI_API_KEY — required for AI requests; set this in Render Environment, never in the Android app

OPENAI_MODEL — optional; defaults to gpt-4o-mini

PORT — optional; hosting platform normally supplies this


Run locally

python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"  # Windows PowerShell: $env:OPENAI_API_KEY="your-key"
uvicorn main:app --host 0.0.0.0 --port 8000

Open http://127.0.0.1:8000/docs to inspect the API.


Android configuration

Set the Android app's backend base URL to the Render service root, for example:


https://YOUR-SERVICE.onrender.com


Use these endpoint paths:


Setting	Path
Chat	/api/chat
Health	/api/health
Call event	/api/call/event
Transcript summary	/api/call/transcript

The Android chat request accepts message, optional session_id, optional language, and optional history (only user and assistant roles are accepted). It returns reply, session_id, and intent.


Important telephony limitation

The call-event route is an API acknowledgement only. It does not receive cellular call audio, answer/reject a call, or send synthesized speech into a cellular call. Those functions depend on Android Telecom roles, device/OEM capabilities, and/or a supported VoIP or programmable-telephony integration. Do not treat an event acknowledgement or generated transcript summary as proof of a live AI phone conversation.


Privacy and session behavior

Chat history is supplied by the Android app per request. The backend does not keep one shared global conversation list, so separate Android sessions do not leak context into each other. Call transcripts are processed only when the app explicitly submits them to /api/call/transcript; this backend version does not persist call records.


Do not commit .env, API keys, access tokens, or private caller data to source control.
