# LLM Quiz Bot

This repository implements the **LLM Analysis Quiz** endpoint required for the TDS project.  
It provides a FastAPI server that accepts quiz tasks (POST `/quiz`), renders JS-heavy quiz pages using Playwright, downloads files (PDF/CSV/JSON), computes answers, and submits them to the quiz submit URL. The server supports multi-step quizzes (follows next `url` returned by submit responses) and respects the 3-minute time limit.

---

## Features

- FastAPI server with `/quiz` POST endpoint.
- Uses Playwright (Chromium) to render dynamic JS pages.
- Downloads and parses PDF/CSV/JSON files; sums `"value"` columns by default.
- Produces base64 PNG charts as fallback answers.
- Automatically follows next quiz URLs until completion or time budget exhausted.
- Loads configuration from `.env`.

---

## Files

- `app.py` — main FastAPI app and solver logic.
- `Dockerfile` — production Dockerfile with Playwright support.
- `requirements.txt` — Python dependencies.
- `.env` — local environment file (not committed).
- `render.yaml` — optional Render deployment manifest.

---

## Requirements

- Python 3.10+ (for local development)
- Docker (for containerized runs)
- Playwright (installed automatically inside Docker)
- `python -m playwright install` (for local dev)

---

## Setup (local, development)

1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd llm-quiz-bot
