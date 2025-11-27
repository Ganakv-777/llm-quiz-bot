# app.py
import os
import json
import re
import base64
import tempfile
import time
import asyncio
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import httpx
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import logging

# ---------- config ----------
load_dotenv()

EXPECTED_SECRET = os.getenv("SECRET") or "gkv"
AI_PIPE_KEY = os.getenv("AI_PIPE_KEY")
API_URL = os.getenv("API_URL")  # AI Pipe endpoint, e.g. https://api.example.com/v1/chat
MODEL = os.getenv("MODEL") or "gpt-5-nano"
QUIZ_EMAIL = os.getenv("QUIZ_EMAIL") or "your-email@domain.com"

USER_AGENT = "LLM-Quiz-Bot/1.0"

# Time budget per incoming request in seconds (must be < 180 on some hosts)
TOTAL_TIME_BUDGET = int(os.getenv("TOTAL_TIME_BUDGET", "170"))

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-quiz-bot")

app = FastAPI(title="LLM Quiz Bot")

# ---------- request model ----------
class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str

# ---------- helpers: rendering / download / parsing ----------
async def render_page(url: str, timeout_ms: int = 60_000) -> Dict[str, Any]:
    """
    Render a page with Playwright and return text/html/links/form actions/pre tags.
    """
    logger.info("Rendering page: %s", url)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox"])
            context = await browser.new_context(user_agent=USER_AGENT)
            page = await context.new_page()
            try:
                await page.goto(url, timeout=timeout_ms)
                await page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except PWTimeoutError:
                # sometimes pages take longer; capture what we can
                logger.warning("Playwright timeout while loading %s", url)

            html = await page.content()
            try:
                text = await page.inner_text("body")
            except:
                text = ""

            # collect links
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
            return {"html": html, "text": text, "links": links, "pres": pres, "form_actions": form_actions}
    except Exception as e:
        logger.exception("render_page error: %s", e)
        return {"html": "", "text": "", "links": [], "pres": [], "form_actions": []}

def extract_json_from_texts(texts: List[str]) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object embedded in text (e.g., inside <pre>).
    """
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

def download_file(url: str, tmpdir: str, timeout: int = 60) -> Optional[str]:
    """
    Download file synchronously (small files) using httpx.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            if r.status_code == 200:
                fname = url.split("?")[0].split("/")[-1] or "file"
                path = os.path.join(tmpdir, fname)
                with open(path, "wb") as f:
                    f.write(r.content)
                return path
            else:
                logger.warning("download_file status %s for %s", r.status_code, url)
    except Exception as e:
        logger.exception("download_file error for %s: %s", url, e)
    return None

def parse_pdf_sum(pdf_path: str, page_no=2, column_name="value") -> Optional[float]:
    """
    Try to parse a table from a PDF and sum a numeric column.
    """
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
    except Exception as e:
        logger.debug("PDF parse error: %s", e)
    return None

def parse_csv_sum(csv_path: str, column_name="value") -> Optional[float]:
    try:
        df = pd.read_csv(csv_path)
        if column_name in df.columns:
            return df[column_name].apply(pd.to_numeric, errors="coerce").sum()
    except Exception as e:
        logger.debug("CSV parse error: %s", e)
    return None

def make_plot_base64(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    plt.figure()
    df.plot(x=x_col, y=y_col, legend=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name)
    plt.close()
    with open(tmp.name, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b}"

# ---------- AI pipe integration ----------
async def call_ai_pipe_extract(page_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask AI Pipe to parse the page snapshot and return structured instructions:
    {
      "submit_url": "... or null",
      "method": "POST" or "GET",
      "headers": {"Content-Type":"application/json"},
      "payload": {"answer": "...", ...},
      "explain": "short explanation"
    }
    """
    if not AI_PIPE_KEY or not API_URL:
        raise RuntimeError("AI_PIPE_KEY or API_URL not set in environment")

    prompt = {
        "role": "user",
        "content": (
            "You are an assistant to parse an exam/quiz webpage and decide how to submit an answer.\n\n"
            "I will provide a JSON snapshot of the page with fields: text, html, links, pres, form_actions.\n"
            "Return a JSON object with keys: submit_url (or null), method (POST/GET), headers (object), payload (object), explain (string).\n\n"
            "Rules:\n"
            " - If you find an explicit JSON object containing 'submit' or 'submit_url' or 'submitUrl', use that submit_url.\n"
            " - If a form action with absolute http/https exists, prefer that.\n"
            " - If there's a link that looks like an API endpoint (/submit,/answer,/post/...), prefer it.\n"
            " - Build payload as {email: <email>, secret: <secret>, url: <current page url>, answer: <computed answer>} when the page expects an 'answer'.\n"
            " - If the page requires file upload or non-JSON submission, return submit_url and explain, but do not attempt binary upload.\n"
            " - If you cannot find a submit URL, return submit_url=null.\n\n"
            "Now parse this snapshot and return the decision JSON only (no extra text):\n\n"
            + json.dumps(page_snapshot) + "\n"
        )
    }

    # Build body for AI pipe - adjust depending on AI Pipe API spec
    body = {
        "model": MODEL,
        "messages": [prompt],
        "temperature": 0.0,
        "max_tokens": 800
    }

    headers = {"Authorization": f"Bearer {AI_PIPE_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(API_URL, headers=headers, json=body)
        try:
            data = r.json()
        except Exception:
            text = r.text
            logger.error("AI pipe returned non-json: %s", text)
            raise RuntimeError("AI pipe error: non-json response")

    # The response format depends on AI Pipe; attempt to get assistant content heuristically
    # Common pattern: {"choices":[{"message":{"content":"{...json...}"}}]}
    content = None
    if isinstance(data, dict):
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            c = data["choices"][0]
            if isinstance(c, dict) and "message" in c and isinstance(c["message"], dict):
                content = c["message"].get("content")
        # fallback for other shapes:
        if content is None and "output" in data:
            content = data.get("output")
    if content is None:
        # try top-level 'text'
        content = data.get("text") if isinstance(data, dict) else None

    if not content:
        raise RuntimeError("AI pipe returned no content")

    # attempt to find JSON object inside content
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start:end+1])
            return parsed
        except Exception:
            # if not valid JSON, return raw content in explain
            return {"submit_url": None, "method": "POST", "headers": {"Content-Type":"application/json"}, "payload": {}, "explain": content}

    # if not JSON, return explain
    return {"submit_url": None, "method": "POST", "headers": {"Content-Type":"application/json"}, "payload": {}, "explain": content}

# ---------- HTTP helper ----------
async def post_json(url: str, payload: Dict[str,Any], headers: Optional[Dict[str,str]] = None, timeout=60.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers=headers or {})
        try:
            return r.json()
        except:
            return {"status_code": r.status_code, "text": r.text}

# ---------- core solver flow ----------
async def solve_quiz_flow(start_url: str, email: str, secret: str, time_budget: float):
    trace = []
    deadline = time.monotonic() + time_budget
    current_url = start_url
    visited = 0
    MAX_VISITS = 40

    while current_url and time.monotonic() < deadline and visited < MAX_VISITS:
        visited += 1
        step = {"step": visited, "url": current_url, "start_time": time.time(), "actions": []}

        try:
            # render
            render = await render_page(current_url)
            step["actions"].append("rendered")

            # prepare snapshot for AI
            snapshot = {
                "url": current_url,
                "text": render.get("text", "")[:50_000],  # limit size
                "links": render.get("links", [])[:200],
                "form_actions": render.get("form_actions", [])[:50],
                "pres": render.get("pres", [])[:10]
            }

            # call AI pipe to extract submit instructions
            try:
                decision = await call_ai_pipe_extract(snapshot)
                step["ai_decision"] = decision
                step["actions"].append("ai_parsed")
            except Exception as e:
                logger.exception("AI parsing failed: %s", e)
                # fallback heuristics (simple)
                decision = {"submit_url": None, "method": "POST", "headers": {"Content-Type":"application/json"}, "payload": {}, "explain": f"ai_error:{e}"}

            # determine the answer using simple heuristics and downloaded files if needed
            answer = None
            tmpdir = tempfile.mkdtemp()
            downloaded = []
            # search for downloadable link in candidate JSON in pres
            candidate_json = extract_json_from_texts(render.get("pres", []))
            download_url = None
            if candidate_json:
                for key in ("file","download","url","data_url"):
                    if key in candidate_json and isinstance(candidate_json[key], str) and candidate_json[key].startswith("http"):
                        download_url = candidate_json[key]; break

            # fallback to links list that end in .pdf/.csv/.json
            if not download_url:
                for l in render.get("links", []):
                    if isinstance(l, str) and any(l.lower().endswith(ext) for ext in [".pdf", ".csv", ".json"]):
                        download_url = l; break

            if download_url:
                logger.info("Downloading file for analysis: %s", download_url)
                path = download_file(download_url, tmpdir)
                if path:
                    downloaded.append(path)
                    step["actions"].append(f"downloaded:{os.path.basename(path)}")

            # try parse downloaded files
            for p in downloaded:
                if p.lower().endswith(".pdf") and answer is None:
                    s = parse_pdf_sum(p)
                    if s is not None:
                        answer = s; step["actions"].append("pdf_parsed")
                if p.lower().endswith(".csv") and answer is None:
                    s = parse_csv_sum(p)
                    if s is not None:
                        answer = s; step["actions"].append("csv_parsed")
                if p.lower().endswith(".json") and answer is None:
                    try:
                        with open(p) as f:
                            j = json.load(f)
                            if isinstance(j, list):
                                vals = [float(x.get("value",0)) for x in j if isinstance(x, dict)]
                                answer = sum(vals); step["actions"].append("json_parsed")
                    except Exception:
                        pass

            # fallback to candidate JSON answer field
            if answer is None and candidate_json and "answer" in candidate_json:
                answer = candidate_json["answer"]; step["actions"].append("used_candidate_answer")

            # otherwise ask AI to compute the answer (give the snapshot + instruction)
            if answer is None:
                # Ask AI: "what is the answer to this page? produce a short answer string or structured JSON"
                req_for_ai = {
                    "url": current_url,
                    "text_sample": snapshot["text"][:4000],
                    "pres": snapshot["pres"][:3],
                    "links": snapshot["links"][:10],
                }
                # We ask the AI to produce a short answer string (or JSON with 'answer' field)
                try:
                    ai_for_answer = {
                        "role": "user",
                        "content": (
                            "Given this page snapshot, return a JSON with {'answer': <value or string>, 'explain': <one-line>}. "
                            "If multiple possible answers exist, return the most plausible answer. "
                            "Snapshot: " + json.dumps(req_for_ai)
                        )
                    }
                    body = {"model": MODEL, "messages": [ai_for_answer], "temperature": 0.0, "max_tokens": 300}
                    headers = {"Authorization": f"Bearer {AI_PIPE_KEY}", "Content-Type": "application/json"}
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        r = await client.post(API_URL, headers=headers, json=body)
                        data = r.json()
                    # extract content
                    content = None
                    if "choices" in data and len(data["choices"])>0:
                        c = data["choices"][0]
                        if "message" in c:
                            content = c["message"].get("content")
                    if not content:
                        # try fallback
                        content = data.get("output") or data.get("text") or str(data)
                    # try to parse answer JSON from content
                    mstart = content.find("{")
                    mend = content.rfind("}")
                    if mstart != -1 and mend != -1 and mend > mstart:
                        try:
                            j = json.loads(content[mstart:mend+1])
                            if isinstance(j, dict) and "answer" in j:
                                answer = j["answer"]
                                step["actions"].append("ai_answer_json")
                            else:
                                # maybe content itself is the answer string
                                answer = content.strip()
                                step["actions"].append("ai_answer_text")
                        except Exception:
                            answer = content.strip(); step["actions"].append("ai_answer_text")
                    else:
                        answer = content.strip(); step["actions"].append("ai_answer_text")
                except Exception as ex:
                    logger.exception("AI answer attempt failed: %s", ex)
                    # last resort fallback
                    answer = "anything you want"
                    step["actions"].append("fallback_answer_used")

            step["computed_answer_summary"] = answer if isinstance(answer, (str,int,float)) else "binary_or_large_payload"

            # Prepare submission per decision from AI parser
            submit_url = decision.get("submit_url") if isinstance(decision, dict) else None
            method = (decision.get("method") if isinstance(decision, dict) else "POST") or "POST"
            headers = decision.get("headers") if isinstance(decision, dict) else {"Content-Type":"application/json"}
            payload_template = decision.get("payload") if isinstance(decision, dict) else {}

            # Fill payload with standard fields if not present
            payload = dict(payload_template) if isinstance(payload_template, dict) else {}
            # set common fields
            if "email" not in payload:
                payload["email"] = email
            if "secret" not in payload:
                payload["secret"] = secret
            if "url" not in payload:
                payload["url"] = current_url
            # set answer field intelligently
            if "answer" not in payload:
                payload["answer"] = answer

            step["prepared_submission"] = {"submit_url": submit_url, "method": method, "headers": headers, "payload": payload}

            submission_result = None
            if submit_url:
                # perform post
                try:
                    if method.upper() == "POST":
                        submission_result = await post_json(submit_url, payload, headers=headers)
                    else:
                        # GET with query params
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            r = await client.get(submit_url, params=payload, headers=headers)
                            try:
                                submission_result = r.json()
                            except:
                                submission_result = {"status_code": r.status_code, "text": r.text}
                    step["actions"].append("posted")
                    step["submission_result"] = submission_result
                except Exception as e:
                    logger.exception("Submission failed: %s", e)
                    step["actions"].append(f"post_error:{str(e)[:200]}")
            else:
                step["actions"].append("no_submit_url_found")

            step["end_time"] = time.time()
            trace.append(step)

            # decide next url
            next_url = None
            if isinstance(submission_result, dict):
                # common pattern: server returns {"url":"..."}
                next_url = submission_result.get("url")
                # sometimes in 'redirect' or 'next'
                if not next_url:
                    next_url = submission_result.get("next") or submission_result.get("redirect")
                if not next_url:
                    # search text for http
                    reason = submission_result.get("reason") or json.dumps(submission_result)
                    m = re.search(r"https?://[^\s'\"<>]+", str(reason))
                    if m:
                        next_url = m.group(0)

            # fallback: search page links for something with "quiz" or "next"
            if not next_url:
                for l in render.get("links", []):
                    if isinstance(l, str) and ("quiz" in l.lower() or "next" in l.lower()) and l.startswith("http"):
                        next_url = l; break

            if next_url:
                current_url = next_url
                logger.info("Following next_url: %s", next_url)
            else:
                # finished
                break

        except Exception as e:
            logger.exception("solve loop exception: %s", e)
            step["error"] = str(e)
            step["end_time"] = time.time()
            trace.append(step)
            break

    return {"trace": trace, "time_left": max(0, deadline - time.monotonic())}

# ---------- FastAPI endpoint ----------
@app.post("/quiz")
async def quiz_endpoint(request: Request):
    # Validate request body
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        payload = QuizPayload(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad fields: {e}")

    if payload.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    start_time = time.monotonic()
    try:
        result = await solve_quiz_flow(payload.url, payload.email, payload.secret, TOTAL_TIME_BUDGET)
        total_elapsed = time.monotonic() - start_time
        result["total_elapsed_seconds"] = total_elapsed
        result["status"] = "completed"
        return result
    except Exception as e:
        logger.exception("quiz_endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------- simple root ----------
@app.get("/")
def home():
    return {"message": "working"}

# ---------- run with uvicorn if executed directly ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
