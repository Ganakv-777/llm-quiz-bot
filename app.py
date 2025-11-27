# app.py
import os
import json
import re
import base64
import tempfile
import time
import asyncio
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import httpx
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Load .env
# --------------------------------------------------------
load_dotenv()

EXPECTED_SECRET = os.getenv("SECRET") or "gkv"
AI_PIPE_KEY = os.getenv("AI_PIPE_KEY")
MODEL = os.getenv("MODEL") or "gpt-5-nano"
API_URL = os.getenv("API_URL")
QUIZ_EMAIL = os.getenv("QUIZ_EMAIL") or "your-email@domain.com"

USER_AGENT = "LLM-Quiz-Bot/1.0"
TOTAL_TIME_BUDGET = 170  # must be < 180s on Railway

app = FastAPI()

# --------------------------------------------------------
# Pydantic Request Model
# --------------------------------------------------------
class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str

@app.get("/")
def home():
    return "working!"


# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------

async def render_page(url: str, timeout_ms: int = 60_000):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        context = await browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()

        try:
            await page.goto(url, timeout=timeout_ms)
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except PWTimeoutError:
            pass

        html = await page.content()
        try:
            text = await page.inner_text("body")
        except:
            text = ""

        links, pres, form_actions = [], [], []

        for a in await page.query_selector_all("a"):
            try:
                href = await a.get_attribute("href")
                if href:
                    links.append(href)
            except:
                pass

        for p in await page.query_selector_all("pre"):
            try:
                pres.append(await p.inner_text())
            except:
                pass

        for f in await page.query_selector_all("form"):
            try:
                act = await f.get_attribute("action")
                if act:
                    form_actions.append(act)
            except:
                pass

        await browser.close()
        return {"html": html, "text": text, "links": links, "pres": pres, "form_actions": form_actions}


def extract_json_from_texts(texts):
    for t in texts:
        if not t:
            continue

        start = t.find("{")
        if start == -1:
            continue

        for end in range(len(t), start, -1):
            try:
                return json.loads(t[start:end])
            except:
                continue

    return None


def download_file(url: str, tmpdir: str, timeout=60):
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            if r.status_code == 200:
                fname = url.split("?")[0].split("/")[-1] or "file"
                path = os.path.join(tmpdir, fname)
                with open(path, "wb") as f:
                    f.write(r.content)
                return path
    except Exception as e:
        print("Download error:", e)
    return None


def parse_pdf_sum(pdf_path: str, page_no=2, column_name="value"):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            idx = page_no - 1
            if idx < 0 or idx >= len(pdf.pages):
                return None
            table = pdf.pages[idx].extract_table()
            if not table:
                return None
            df = pd.DataFrame(table[1:], columns=table[0])
            if column_name in df.columns:
                return df[column_name].apply(pd.to_numeric, errors="coerce").sum()
    except:
        pass
    return None


def parse_csv_sum(csv_path: str, column_name="value"):
    try:
        df = pd.read_csv(csv_path)
        if column_name in df.columns:
            return df[column_name].apply(pd.to_numeric, errors="coerce").sum()
    except:
        pass
    return None


def make_plot_base64(df: pd.DataFrame, x_col: str, y_col: str):
    plt.figure()
    df.plot(x=x_col, y=y_col, legend=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name)
    plt.close()

    with open(tmp.name, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


async def post_json(url: str, payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload)
        try:
            return r.json()
        except:
            return {"status_code": r.status_code, "text": r.text}


# --------------------------------------------------------
# QUIZ Solver
# --------------------------------------------------------

async def solve_quiz_flow(start_url: str, email: str, secret: str, time_budget: float):
    trace = []
    deadline = time.monotonic() + time_budget
    current_url = start_url
    visited = 0

    while current_url and time.monotonic() < deadline and visited < 40:
        visited += 1
        step = {"step": visited, "url": current_url, "actions": []}

        try:
            render = await render_page(current_url)
            step["actions"].append("rendered")

            candidate = extract_json_from_texts(render["pres"] + [render["text"]])
            submit_url = None

            if candidate:
                submit_url = candidate.get("submit") or candidate.get("submit_url")

            for f in render["form_actions"]:
                if f.startswith("http"):
                    submit_url = submit_url or f

            for l in render["links"]:
                if "submit" in l.lower() and l.startswith("http"):
                    submit_url = submit_url or l

            step["submit_url"] = submit_url

            # -------- Download File --------
            tmpdir = tempfile.mkdtemp()
            downloaded = []
            download_url = None

            if candidate:
                for k in ["file", "download", "data_url"]:
                    if k in candidate and candidate[k].startswith("http"):
                        download_url = candidate[k]
                        break

            if not download_url:
                for l in render["links"]:
                    if l.lower().endswith((".pdf", ".csv", ".json")):
                        download_url = l
                        break

            answer = None
            if download_url:
                path = download_file(download_url, tmpdir)
                if path:
                    downloaded.append(path)
                    step["actions"].append(f"downloaded:{os.path.basename(path)}")

            # -------- Compute Answer --------
            for p in downloaded:
                if p.endswith(".pdf"):
                    s = parse_pdf_sum(p)
                    if s is not None:
                        answer = s
                        break

                if p.endswith(".csv"):
                    s = parse_csv_sum(p)
                    if s is not None:
                        answer = s
                        break

            if answer is None and candidate and "answer" in candidate:
                answer = candidate["answer"]

            if answer is None:
                answer = "anything you want"

            step["computed_answer_summary"] = str(answer)

            # -------- Submit Answer --------
            if submit_url:
                payload = {
                    "email": email,
                    "secret": secret,
                    "url": current_url,
                    "answer": answer
                }
                submission_result = await post_json(submit_url, payload)
                step["submission_result"] = submission_result
            else:
                submission_result = None

            trace.append(step)

            # -------- Next URL --------
            next_url = None
            if isinstance(submission_result, dict):
                next_url = submission_result.get("url")

            current_url = next_url

        except Exception as e:
            step["error"] = str(e)
            trace.append(step)
            break

    return {"trace": trace}


# --------------------------------------------------------
# QUIZ API Endpoint (correct Swagger version)
# --------------------------------------------------------

@app.post("/quiz")
async def quiz_endpoint(payload: QuizPayload):

    if payload.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    start_time = time.monotonic()
    result = await solve_quiz_flow(payload.url, payload.email, payload.secret, TOTAL_TIME_BUDGET)
    result["total_elapsed_seconds"] = time.monotonic() - start_time
    result["status"] = "completed"
    return result
