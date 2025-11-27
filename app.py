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


# ============================================================
# Load environment variables
# ============================================================
load_dotenv()

EXPECTED_SECRET = os.getenv("SECRET") or "gkv"
AI_PIPE_KEY = os.getenv("AI_PIPE_KEY")
API_URL = os.getenv("API_URL")
MODEL = os.getenv("MODEL") or "gpt-5-nano"

USER_AGENT = "LLM-Quiz-Bot/1.0"
TOTAL_TIME_BUDGET = 170   # < 180 sec for Railway


# ============================================================
# FastAPI application
# ============================================================
app = FastAPI(title="LLM Quiz Bot", version="1.0.0")


# ============================================================
# Request Body for /quiz
# ============================================================
class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str


# ============================================================
# Home endpoint
# ============================================================
@app.get("/")
def home():
    return "working"


# ============================================================
# Helper: Render page using Playwright
# ============================================================
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

        # extract links, <pre> blocks, form actions
        links = []
        for a in await page.query_selector_all("a"):
            try:
                href = await a.get_attribute("href")
                if href:
                    links.append(href)
            except:
                pass

        pres = []
        for p in await page.query_selector_all("pre"):
            try:
                pres.append(await p.inner_text())
            except:
                pass

        form_actions = []
        for f in await page.query_selector_all("form"):
            try:
                act = await f.get_attribute("action")
                if act:
                    form_actions.append(act)
            except:
                pass

        await browser.close()

        return {
            "html": html,
            "text": text,
            "links": links,
            "pres": pres,
            "form_actions": form_actions,
        }


# ============================================================
# Helper: Extract embedded JSON
# ============================================================
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


# ============================================================
# Helper: Download file (CSV/PDF)
# ============================================================
def download_file(url: str, tmpdir: str, timeout: int = 60):
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            if r.status_code == 200:
                filename = url.split("?")[0].split("/")[-1] or "file"
                path = os.path.join(tmpdir, filename)
                with open(path, "wb") as f:
                    f.write(r.content)
                return path
    except:
        return None
    return None


# ============================================================
# PDF / CSV helpers
# ============================================================
def parse_pdf_sum(pdf_path: str, page_no=2, column_name="value"):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_no - 1]
            table = page.extract_table()
            if not table:
                return None
            df = pd.DataFrame(table[1:], columns=table[0])
            if column_name in df.columns:
                return df[column_name].apply(pd.to_numeric, errors="coerce").sum()
    except:
        return None
    return None


def parse_csv_sum(csv_path: str, column_name="value"):
    try:
        df = pd.read_csv(csv_path)
        if column_name in df.columns:
            return df[column_name].apply(pd.to_numeric, errors="coerce").sum()
    except:
        return None
    return None


# ============================================================
# Make a base64 plot when quiz needs a graph
# ============================================================
def make_plot_base64(df: pd.DataFrame, x_col: str, y_col: str):
    plt.figure()
    df.plot(x=x_col, y=y_col, legend=False)

    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp.name)
    plt.close()

    with open(temp.name, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"


# ============================================================
# Helper: POST JSON
# ============================================================
async def post_json(url: str, payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload)
        try:
            return r.json()
        except:
            return {"status_code": r.status_code, "text": r.text}


# ============================================================
# Optional: Call AI Pipe
# ============================================================
async def call_ai_pipe(prompt: str):
    if not AI_PIPE_KEY or not API_URL:
        raise RuntimeError("AI_PIPE_KEY or API_URL missing")

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {AI_PIPE_KEY}"}
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}

        r = await client.post(API_URL, headers=headers, json=body)
        data = r.json()
        return data["choices"][0]["message"]["content"]


# ============================================================
# MAIN: Solve quiz step-by-step
# ============================================================
async def solve_quiz_flow(start_url: str, email: str, secret: str, time_budget: float):
    trace = []
    current_url = start_url
    deadline = time.monotonic() + time_budget
    visited = 0
    MAX_VISITS = 40

    while current_url and time.monotonic() < deadline and visited < MAX_VISITS:
        visited += 1
        step = {"step": visited, "url": current_url, "actions": []}

        try:
            render = await render_page(current_url)
            step["actions"].append("rendered")

            # detect embedded JSON
            candidate = extract_json_from_texts(render["pres"] + [render["text"]])

            submit_url = None
            if candidate:
                submit_url = candidate.get("submit") or candidate.get("submit_url")

            # fallback to forms
            for f in render["form_actions"]:
                if f.startswith("http"):
                    submit_url = submit_url or f

            # fallback to links
            for l in render["links"]:
                if "submit" in l.lower() and l.startswith("http"):
                    submit_url = submit_url or l

            step["submit_url"] = submit_url

            # download files
            tmpdir = tempfile.mkdtemp()
            downloaded = []
            download_url = None

            if candidate:
                for key in ["file", "download", "data_url"]:
                    if key in candidate and str(candidate[key]).startswith("http"):
                        download_url = candidate[key]

            if not download_url:
                for l in render["links"]:
                    if any(ext in l.lower() for ext in [".csv", ".pdf", ".json"]):
                        download_url = l
                        break

            if download_url:
                p = download_file(download_url, tmpdir)
                if p:
                    downloaded.append(p)

            # compute answer
            answer = None
            for fp in downloaded:
                if fp.endswith(".pdf"):
                    answer = parse_pdf_sum(fp)
                    if answer is not None:
                        break
                if fp.endswith(".csv"):
                    answer = parse_csv_sum(fp)
                    if answer is not None:
                        break
                if fp.endswith(".json"):
                    try:
                        with open(fp) as f:
                            j = json.load(f)
                        answer = sum(float(x["value"]) for x in j)
                        break
                    except:
                        pass

            if answer is None and candidate and "answer" in candidate:
                answer = candidate["answer"]

            if answer is None:
                rows = []
                for ln in render["text"].splitlines():
                    if re.match(r"^\w+,\s*\d+(\.\d+)?$", ln):
                        name, val = ln.split(",")
                        rows.append((name.strip(), float(val)))
                if rows:
                    df = pd.DataFrame(rows, columns=["x", "y"])
                    answer = make_plot_base64(df, "x", "y")

            if answer is None:
                answer = "anything you want"

            step["computed_answer_summary"] = answer if isinstance(answer, (str, int, float)) else "binary answer"

            # submit
            result = None
            if submit_url:
                payload = {
                    "email": email,
                    "secret": secret,
                    "url": current_url,
                    "answer": answer,
                }
                result = await post_json(submit_url, payload)
                step["submission_result"] = result

                if isinstance(result, dict) and result.get("url"):
                    current_url = result["url"]
                    trace.append(step)
                    continue

            trace.append(step)
            break

        except Exception as e:
            step["error"] = str(e)
            trace.append(step)
            break

    return {"trace": trace}


# ============================================================
# FINAL API ENDPOINT — THIS is the correct one
# ============================================================
@app.post("/quiz")
async def quiz_endpoint(payload: QuizPayload):
    if payload.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    start = time.monotonic()
    result = await solve_quiz_flow(payload.url, payload.email, payload.secret, TOTAL_TIME_BUDGET)
    result["total_elapsed_seconds"] = time.monotonic() - start
    result["status"] = "completed"
    return result
