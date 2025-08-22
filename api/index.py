from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import ast
import operator
import requests
import json
import re
import threading
import uuid
from pathlib import Path

# This is a simple Flask server that serves a JSON response
app = Flask(__name__)
CORS(app)

# Load variables from .env if present
load_dotenv()

# Configure Gemini API from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Choose a lightweight default model; can be changed to gemini-1.5-pro if desired
    GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
else:
    GEMINI_MODEL = None

# JSON Chat DB config
CHAT_DB_PATH = Path(__file__).parent / "chats.json"
CHAT_DB_LOCK = threading.Lock()

def _ensure_chat_db():
    with CHAT_DB_LOCK:
        if not CHAT_DB_PATH.exists():
            CHAT_DB_PATH.write_text(json.dumps({"chats": []}, ensure_ascii=False, indent=2), encoding="utf-8")

def _read_chat_db() -> dict:
    _ensure_chat_db()
    with CHAT_DB_LOCK:
        try:
            return json.loads(CHAT_DB_PATH.read_text(encoding="utf-8") or "{}") or {"chats": []}
        except Exception:
            return {"chats": []}

def _write_chat_db(data: dict) -> None:
    with CHAT_DB_LOCK:
        CHAT_DB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "Like this video if this helped!",
        'people': ['Jack', 'Harry', 'Arpan']
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Chat endpoint backed by Google Gemini.

    Request JSON: { "message": string, "history": [{"role":"user|model","parts":[{"text": string}]}]? }
    Response JSON: { "reply": string }
    """
    if GEMINI_MODEL is None:
        return jsonify({
            "error": "Server not configured. Please set GEMINI_API_KEY environment variable."
        }), 500

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []
    agentic = bool(data.get("agentic", False))
    model_name = (data.get("model") or "").strip() or "gemini-1.5-flash"
    temperature = data.get("temperature")
    system_prompt = (data.get("system_prompt") or "").strip()
    max_steps = data.get("max_steps")
    try:
        max_steps = int(max_steps) if max_steps is not None else 3
    except Exception:
        max_steps = 3

    if not message:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    try:
        # Prepare model (allow override)
        model = GEMINI_MODEL if model_name == "gemini-1.5-flash" else genai.GenerativeModel(model_name)

        gen_config = {}
        if isinstance(temperature, (int, float)):
            gen_config["temperature"] = float(temperature)

        # Agentic multi-step tool usage with trace
        if agentic:
            instruction = (
                (system_prompt + "\n\n") if system_prompt else ""
            ) + (
                "You are an agent that may call tools in order to answer. "
                "Available tools:\n"
                "- math: evaluate arithmetic expression, input JSON: {\"expression\": string}\n"
                "- fetch_url: fetch a URL and return text, input JSON: {\"url\": string}\n"
                "- youtube_search: find YouTube videos, input JSON: {\"query\": string, \"max_results\"?: number, \"order\"?: string}\n\n"
                "At each step, respond ONLY as strict JSON with keys: action: 'tool'|'final', "
                "tool_name (if action=='tool'), tool_input (object, if action=='tool'), answer (if action=='final')."
            )

            agent_history = [
                {"role": "user", "parts": [{"text": instruction}]},
                *history,
                {"role": "user", "parts": [{"text": message}]},
            ]

            trace = []
            step = 0
            while step < max_steps:
                step += 1
                step_resp = model.generate_content(agent_history, generation_config=gen_config or None)
                draft = getattr(step_resp, "text", "") or ""

                # Parse JSON plan
                tool_plan = None
                for m in re.finditer(r"\{[\s\S]*\}", draft):
                    try:
                        tool_plan = json.loads(m.group(0))
                        break
                    except Exception:
                        continue

                if not tool_plan:
                    # No JSON, treat as final
                    return jsonify({"reply": draft, "trace": trace})

                action = (tool_plan.get("action") or "").lower()
                if action != "tool":
                    final_answer = tool_plan.get("answer", draft)
                    return jsonify({"reply": final_answer, "trace": trace})

                tool_name = tool_plan.get("tool_name")
                tool_input = tool_plan.get("tool_input") or {}

                result = run_tool(tool_name, tool_input)
                trace.append({
                    "step": step,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_result_preview": (result[:200] + "…") if isinstance(result, str) and len(result) > 200 else result,
                })

                # Add tool thought and result back to context then continue
                agent_history.extend([
                    {"role": "model", "parts": [{"text": draft}]},
                    {"role": "user", "parts": [{"text": f"TOOL RESULT ({tool_name}):\n{result}"}]},
                ])

            # Max steps reached
            return jsonify({"reply": "Reached max tool steps without a final answer.", "trace": trace})
        else:
            # Basic single-turn prompt; if history provided, use a chat session
            if history:
                chat_session = model.start_chat(history=history)
                final_resp = chat_session.send_message(message, generation_config=gen_config or None)
            else:
                final_resp = model.generate_content(message, generation_config=gen_config or None)

            # google-generativeai SDK returns text on .text
            reply_text = getattr(final_resp, "text", None)
        if not reply_text and hasattr(final_resp, "candidates"):
            # Fallback parse in case of SDK changes
            reply_text = final_resp.candidates[0].content.parts[0].text if final_resp.candidates else ""

        return jsonify({"reply": reply_text or ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Tools implementation
# -----------------------------

def run_tool(name: str, payload: dict) -> str:
    name = (name or "").lower().strip()
    try:
        if name == "math":
            expr = str(payload.get("expression", "")).strip()
            return safe_eval_math(expr)
        if name == "fetch_url":
            url = str(payload.get("url", "")).strip()
            return fetch_url_text(url)
        if name == "youtube_search":
            query = str(payload.get("query", "")).strip()
            max_results = int(payload.get("max_results", 5))
            order = str(payload.get("order", "relevance")).strip() or "relevance"
            return youtube_search(query, max_results=max_results, order=order)
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"


def safe_eval_math(expression: str) -> str:
    if not expression:
        return "Empty expression"

    # Allowed operators
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    tree = ast.parse(expression, mode="eval")
    result = _eval(tree.body)
    return str(result)


def fetch_url_text(url: str) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http or https")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    text = r.text
    # Truncate to avoid huge responses
    return text[:4000]


def youtube_search(query: str, max_results: int = 5, order: str = "relevance") -> str:
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY not configured on server")
    if not query:
        raise ValueError("query is required")
    max_results = max(1, min(25, int(max_results)))
    params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "type": "video",
        "q": query,
        "maxResults": max_results,
        "order": order or "relevance",
        "safeSearch": "moderate",
    }
    r = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    items = data.get("items", [])
    results = []
    for it in items:
        vid = (it.get("id") or {}).get("videoId")
        snip = it.get("snippet") or {}
        if not vid:
            continue
        results.append({
            "videoId": vid,
            "title": snip.get("title"),
            "channelTitle": snip.get("channelTitle"),
            "publishedAt": snip.get("publishedAt"),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "thumbnail": ((snip.get("thumbnails") or {}).get("medium") or {}).get("url"),
        })
    return json.dumps(results, ensure_ascii=False)


@app.route("/api/youtube/search", methods=["GET"])
def api_youtube_search():
    q = (request.args.get("q") or "").strip()
    max_results = request.args.get("max_results", 5)
    order = (request.args.get("order") or "relevance").strip()
    try:
        max_results = int(max_results)
    except Exception:
        max_results = 5
    try:
        res = youtube_search(q, max_results=max_results, order=order)
        return jsonify({"results": json.loads(res)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# Chat JSON DB Endpoints
# -----------------------------

@app.route("/api/chats/list", methods=["GET"])
def api_chats_list():
    db = _read_chat_db()
    # Return lite info
    items = [
        {"id": c.get("id"), "title": c.get("title"), "createdAt": c.get("createdAt")}
        for c in (db.get("chats") or [])
    ]
    return jsonify({"items": items})


@app.route("/api/chats/get/<chat_id>", methods=["GET"])
def api_chats_get(chat_id: str):
    db = _read_chat_db()
    for c in db.get("chats") or []:
        if c.get("id") == chat_id:
            return jsonify(c)
    return jsonify({"error": "not found"}), 404


@app.route("/api/chats/save", methods=["POST"])
def api_chats_save():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Untitled").strip() or "Untitled"
    messages = data.get("messages") or []
    meta = data.get("meta") or {}
    if not isinstance(messages, list):
        return jsonify({"error": "messages must be an array"}), 400

    db = _read_chat_db()
    chat_id = str(uuid.uuid4())
    record = {
        "id": chat_id,
        "title": title,
        "createdAt": int(__import__("time").time() * 1000),
        "messages": messages,
        "meta": meta,
    }
    all_chats = db.get("chats") or []
    all_chats.append(record)
    db["chats"] = all_chats
    _write_chat_db(db)
    return jsonify({"id": chat_id})


@app.route("/api/chats/delete/<chat_id>", methods=["DELETE"])
def api_chats_delete(chat_id: str):
    db = _read_chat_db()
    before = len(db.get("chats") or [])
    db["chats"] = [c for c in (db.get("chats") or []) if c.get("id") != chat_id]
    after = len(db.get("chats") or [])
    _write_chat_db(db)
    if after == before:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})
if __name__ == "__main__":
    app.run(debug=True, port=8080)
