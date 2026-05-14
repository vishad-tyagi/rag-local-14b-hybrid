from flask import Flask, Response, jsonify, render_template, request
import json
from pathlib import Path

from auto_chain import AutoQueryService
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
auto_service = None


def get_graph_service():
    global graph_service

    if graph_service is not None:
        return graph_service

    if GraphQueryService is None:
        raise RuntimeError(f"Graph mode is unavailable: {GRAPH_IMPORT_ERROR}")

    graph_service = GraphQueryService()
    return graph_service


def get_auto_service():
    global auto_service

    if auto_service is None:
        auto_service = AutoQueryService(
            rag_chain=rag_chain,
            retriever=retriever,
            llm=llm,
            sql_service=sql_service,
            graph_service_factory=get_graph_service,
        )

    return auto_service


@app.get("/")
def home():
    return render_template("index.html", answer=None, question=None, sources=None)


@app.get("/api/health")
def api_health():
    index_dir = Path("vectorstore/faiss_index")
    db_path = Path("data.db")

    return jsonify({
        "status": "ok",
        "modes": ["rag", "sql", "graph", "auto"],
        "rag": {
            "available": (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists(),
            "index_path": str(index_dir),
        },
        "sql": {
            "available": db_path.exists(),
            "db_path": str(db_path),
        },
        "graph": {
            "import_available": GraphQueryService is not None,
            "import_error": GRAPH_IMPORT_ERROR,
            "connection_checked": False,
        },
    })


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


@app.post("/api/auto/query")
def api_auto_query():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({
            "mode": "auto",
            "selected_modes": [],
            "route": {
                "confidence": 0.0,
                "reason": "Please enter a question.",
                "router": "none",
                "requires_synthesis": False,
            },
            "answer": "Please enter a question.",
            "sources": [],
            "sql": "",
            "columns": [],
            "rows": [],
            "cypher": "",
            "results": [],
            "pipeline_outputs": {},
            "partial_errors": [],
        }), 400

    try:
        return jsonify(get_auto_service().run(question))
    except Exception as e:
        return jsonify({
            "mode": "auto",
            "selected_modes": [],
            "route": {
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
                "router": "error",
                "requires_synthesis": False,
            },
            "answer": f"Error: {str(e)}",
            "sources": [],
            "sql": "",
            "columns": [],
            "rows": [],
            "cypher": "",
            "results": [],
            "pipeline_outputs": {},
            "partial_errors": [{"mode": "auto", "error": str(e)}],
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
