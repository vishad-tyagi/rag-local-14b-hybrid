from flask import Flask, render_template, request, jsonify, Response
import json
import requests

from rag_chain import build_rag

app = Flask(__name__)

rag_chain, retriever = build_rag()

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.1"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say: \"I don't know based on the provided documents.\""
)


@app.get("/")
def home():
    return render_template("index.html", answer=None, question=None, sources=None)


@app.post("/api/ask")
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question.", "sources": []}), 400

    docs = retriever.invoke(question)
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
        }
        for d in docs
    ]

    result = rag_chain.invoke(question)
    answer = getattr(result, "content", str(result))
    return jsonify({"answer": answer, "sources": sources})


@app.post("/api/ask/stream")
def api_ask_stream():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return Response("event: error\ndata: \"Please enter a question.\"\n\n", mimetype="text/event-stream")

    docs = retriever.invoke(question)
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
        }
        for d in docs
    ]

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    def sse():
        # Send sources first
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        try:
            ollama_payload = {
                "model": OLLAMA_MODEL,
                "stream": True,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": 0.2},
            }

            with requests.post(OLLAMA_CHAT_URL, json=ollama_payload, stream=True, timeout=300) as r:
                r.raise_for_status()

                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)

                    if data.get("done"):
                        break

                    delta = (data.get("message") or {}).get("content") or ""
                    if delta:
                        # IMPORTANT: send token as JSON string to preserve spaces/newlines exactly
                        yield f"event: token\ndata: {json.dumps(delta)}\n\n"

            yield "event: done\ndata: \"ok\"\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return Response(sse(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)