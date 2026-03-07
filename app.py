from flask import Flask, Response, jsonify, render_template, request
import json

from rag_chain import SYSTEM_PROMPT, build_prompt, build_rag

app = Flask(__name__)

rag_chain, retriever, llm = build_rag()


@app.get("/")
def home():
    return render_template("index.html", answer=None, question=None, sources=None)


@app.post("/ask")
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        return render_template(
            "index.html",
            answer="Please enter a question.",
            question="",
            sources=[],
        )

    docs = retriever.invoke(question)
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
        }
        for d in docs
    ]

    answer = rag_chain.invoke(question)

    return render_template("index.html", answer=answer, question=question, sources=sources)


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

    answer = rag_chain.invoke(question)
    return jsonify({"answer": answer, "sources": sources})


@app.post("/api/ask/stream")
def api_ask_stream():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return Response(
            'event: error\ndata: "Please enter a question."\n\n',
            mimetype="text/event-stream",
        )

    docs = retriever.invoke(question)
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
        }
        for d in docs
    ]

    prompt = build_prompt(question, docs)

    def sse():
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        try:
            for token in llm.stream_answer(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            ):
                # send token as JSON string so spaces/newlines are preserved
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            yield 'event: done\ndata: "ok"\n\n'

        except Exception as e:
            yield f'event: error\ndata: {json.dumps(str(e))}\n\n'

    return Response(sse(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)