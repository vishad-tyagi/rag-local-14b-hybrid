from flask import Flask, Response, jsonify, render_template, request
import json

from rag_chain import SYSTEM_PROMPT, build_prompt, build_rag
from sql_chain import SQLQueryService

try:
    from graph_chain import GraphQueryService
    GRAPH_IMPORT_ERROR = None
except Exception as exc:
    GraphQueryService = None
    GRAPH_IMPORT_ERROR = str(exc)

app = Flask(__name__)

rag_chain, retriever, llm = build_rag()
sql_service = SQLQueryService()
graph_service = None


def get_graph_service():
    global graph_service

    if graph_service is not None:
        return graph_service

    if GraphQueryService is None:
        raise RuntimeError(f"Graph mode is unavailable: {GRAPH_IMPORT_ERROR}")

    graph_service = GraphQueryService()
    return graph_service


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
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            yield 'event: done\ndata: "ok"\n\n'

        except Exception as e:
            yield f'event: error\ndata: {json.dumps(str(e))}\n\n'

    return Response(sse(), mimetype="text/event-stream")


@app.post("/api/sql/query")
def api_sql_query():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({
            "mode": "sql",
            "summary": "Please enter a question.",
            "sql": "",
            "columns": [],
            "rows": [],
        }), 400

    try:
        result = sql_service.run(question)
        return jsonify({
            "mode": "sql",
            "summary": result["summary"],
            "sql": result["sql"],
            "columns": result["columns"],
            "rows": result["rows"],
        })
    except Exception as e:
        return jsonify({
            "mode": "sql",
            "summary": f"Error: {str(e)}",
            "sql": "",
            "columns": [],
            "rows": [],
        }), 500


@app.post("/api/graph/query")
def api_graph_query():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({
            "mode": "graph",
            "summary": "Please enter a question.",
            "cypher": "",
            "results": []
        }), 400

    try:
        result = get_graph_service().run(question)
        return jsonify({
            "mode": "graph",
            "summary": result["summary"],
            "cypher": result["cypher"],
            "results": result["results"]
        })
    except Exception as e:
        return jsonify({
            "mode": "graph",
            "summary": f"Error: {str(e)}",
            "cypher": "",
            "results": []
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
