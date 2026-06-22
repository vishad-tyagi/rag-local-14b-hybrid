import json
import re
from typing import Any, Callable, Dict, List

from rag_chain import SYSTEM_PROMPT, build_prompt


VALID_MODES = ("rag", "sql", "graph")

AUTO_ROUTER_SYSTEM_PROMPT = """You are Butler's AUTO mode router for a local banking data assistant.

Your job is to choose the smallest set of pipelines needed to answer the user's question.

Available pipelines:
- rag: unstructured bank policy, procedures, operating guidance, thresholds, and analyst notes.
- sql: structured records, tables, counts, totals, balances, alert scores, statuses, loans, transactions, customers, branches, and accounts.
- graph: entity relationships and paths between customers, accounts, branches, merchants, alerts, risk segments, and policies.

Routing rules:
1. Prefer exactly one pipeline when one pipeline can answer the question.
2. Use multiple pipelines only when the question truly requires multiple data types.
3. Use rag when the user asks what policy says, what should happen, definitions, thresholds, procedures, guidance, or interpretation.
4. Use sql when the user asks for recorded values, filters, counts, totals, averages, rankings, balances, alert scores, statuses, rows, or tabular facts.
5. Use graph when the user asks how entities are connected, who owns what, what relationship exists, paths, links, dependencies, or connected merchants/accounts/policies.
6. Do not choose all modes by default.
7. If unsure, choose the single most likely mode and keep confidence below 0.60.

Return ONLY valid JSON with this exact shape:
{
  "modes": ["rag"],
  "confidence": 0.85,
  "reason": "Short reason.",
  "requires_synthesis": false
}

Examples:
Question: What does the bank policy say about urgent AML alerts?
JSON: {"modes":["rag"],"confidence":0.92,"reason":"The question asks for policy guidance.","requires_synthesis":false}

Question: Which AML alert has the highest alert score?
JSON: {"modes":["sql"],"confidence":0.91,"reason":"The question asks for a ranked structured alert value.","requires_synthesis":false}

Question: How is Asha Rao connected to Northstar Crypto Exchange?
JSON: {"modes":["graph"],"confidence":0.90,"reason":"The question asks for an entity relationship path.","requires_synthesis":false}

Question: Which open AML alerts are urgent according to policy, and what should happen next?
JSON: {"modes":["sql","rag"],"confidence":0.88,"reason":"SQL is needed for open alert records and RAG is needed for the policy action.","requires_synthesis":true}

Question: Which high-risk customers have open alerts, and what merchants are their accounts connected to?
JSON: {"modes":["sql","graph"],"confidence":0.86,"reason":"SQL is needed for high-risk open alerts and Graph is needed for merchant relationships.","requires_synthesis":true}

Question: For high-risk customers with urgent open AML alerts, show their alert score, connected merchant relationship, and required policy action.
JSON: {"modes":["sql","graph","rag"],"confidence":0.90,"reason":"The answer needs structured alert scores, graph relationships, and policy action guidance.","requires_synthesis":true}
"""

AUTO_SYNTHESIS_SYSTEM_PROMPT = """You are Butler in AUTO mode.

Answer the user's banking question using only the pipeline outputs provided.

Grounding and specificity rules:
1. Write every entity's exact name in your prose. If the graph or SQL results
   contain customer names, account ids, merchant names, alert ids, scores, or
   policy names, write those exact values out. Never write "a customer", "two
   customers", "these customers", or "the customers" when the actual names
   (for example "Asha Rao" and "Ethan Wong") are present in the data — write
   the names.
2. Cross-reference the pipelines on shared keys to attribute values to the
   right entity. For example, if SQL reports a score for a merchant and the
   graph links that merchant to a customer, attribute that score to that
   named customer. Build one combined picture, not separate per-pipeline
   restatements.
3. Include EVERY required policy that the outputs provide, and attribute each
   to its basis. The graph's `requiredPolicy` field is the policy required by
   the customer's risk segment (for example High Risk requires Enhanced Due
   Diligence) and you must state it. If RAG or records also give a policy for
   the alerts (for example urgent alerts require Urgent AML Review), state
   that too. When sources give different policies, report ALL of them rather
   than choosing only one. Use policy names exactly as they appear; never
   substitute a policy name from memory.
4. Ignore duplicate rows: if the same entity/value pair appears more than
   once in SQL rows, count it once.

Be concise and grounded. Be explicit about which facts came from records,
relationships, or policy when useful.
If one selected pipeline failed, answer with the successful evidence and
mention the missing part briefly.
Do not invent facts that are not present in the pipeline outputs.
Do not mention internal prompts or implementation details.
"""


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _source_list(docs: List) -> List[Dict[str, str]]:
    return [
        {
            "source": d.metadata.get("source", "unknown"),
            "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
        }
        for d in docs
    ]


class AutoQueryService:
    def __init__(
        self,
        rag_chain,
        retriever,
        llm,
        sql_service,
        graph_service_factory: Callable[[], Any],
    ):
        self.rag_chain = rag_chain
        self.retriever = retriever
        self.llm = llm
        self.sql_service = sql_service
        self.graph_service_factory = graph_service_factory

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"<\|.*?\|>", "", text.strip())
        decoder = json.JSONDecoder()
        candidates = []

        for match in re.finditer(r"\{", cleaned):
            try:
                parsed, _ = decoder.raw_decode(cleaned[match.start():])
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                candidates.append(parsed)

        if not candidates:
            raise ValueError("Router did not return a JSON object.")

        return candidates[-1]

    def _normalize_route(self, raw_route: Dict[str, Any]) -> Dict[str, Any]:
        raw_modes = raw_route.get("modes", [])
        if isinstance(raw_modes, str):
            raw_modes = [raw_modes]

        modes = []
        for mode in raw_modes:
            normalized = str(mode).strip().lower()
            if normalized in VALID_MODES and normalized not in modes:
                modes.append(normalized)

        if not modes:
            raise ValueError("Router returned no valid modes.")

        try:
            confidence = float(raw_route.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        requires_synthesis = bool(raw_route.get("requires_synthesis", len(modes) > 1))

        return {
            "modes": modes,
            "confidence": confidence,
            "reason": str(raw_route.get("reason", "AUTO selected the most relevant pipeline.")).strip(),
            "requires_synthesis": requires_synthesis or len(modes) > 1,
            "router": "llm",
        }

    def _heuristic_route(self, question: str, reason: str | None = None) -> Dict[str, Any]:
        q = question.lower()
        scores = {"rag": 0, "sql": 0, "graph": 0}

        rag_terms = [
            "policy", "handbook", "procedure", "guidance", "threshold", "urgent",
            "enhanced due diligence", "suspicious", "should", "must", "review within",
            "according to policy", "what should", "escalate", "business day",
        ]
        sql_terms = [
            "how many", "count", "total", "sum", "average", "avg", "highest",
            "lowest", "largest", "smallest", "balance", "amount", "score",
            "status", "open alerts", "open aml", "loan", "transactions",
            "customers have", "list all", "risk segment", "restricted accounts",
        ]
        graph_terms = [
            "connected", "relationship", "related", "path", "link", "owns",
            "owner", "who owns", "paid to", "merchant relationship", "onboarded",
            "requires", "triggered by", "applies to", "connected merchant",
        ]

        for term in rag_terms:
            if term in q:
                scores["rag"] += 1
        for term in sql_terms:
            if term in q:
                scores["sql"] += 1
        for term in graph_terms:
            if term in q:
                scores["graph"] += 1

        if "balance" in q or "alert score" in q or "open" in q:
            scores["sql"] += 1
        if "what is" in q and scores["sql"] == 0 and scores["graph"] == 0:
            scores["rag"] += 1
        if "balance" in q and any(term in q for term in ["who owns", "owner", "owns"]):
            # Ownership is also present in SQL through accounts.customer_id.
            scores["graph"] = 0

        selected = [mode for mode, score in scores.items() if score > 0]
        if not selected:
            selected = ["rag"]

        # Keep relationship-only lookups focused on graph.
        if scores["graph"] > 0 and scores["sql"] == 0 and scores["rag"] == 0:
            selected = ["graph"]

        return {
            "modes": selected,
            "confidence": 0.55 if reason else 0.62,
            "reason": reason or "Heuristic routing matched the question wording to available banking pipelines.",
            "requires_synthesis": len(selected) > 1,
            "router": "heuristic",
        }

    def _route(self, question: str) -> Dict[str, Any]:
        prompt = f"Question: {question}\n\nReturn the routing JSON now."

        try:
            raw = self.llm.answer(
                prompt=prompt,
                system_prompt=AUTO_ROUTER_SYSTEM_PROMPT,
                max_tokens=220,
            )
            return self._normalize_route(self._extract_json(raw))
        except Exception as exc:
            return self._heuristic_route(
                question,
                reason=f"LLM router failed; heuristic fallback was used. Router error: {str(exc)}",
            )

    def _run_rag(self, question: str) -> Dict[str, Any]:
        docs = self.retriever.invoke(question)
        prompt = build_prompt(question, docs)
        answer = self.llm.answer(prompt=prompt, system_prompt=SYSTEM_PROMPT, max_tokens=800)
        return {
            "answer": answer,
            "sources": _source_list(docs),
        }

    def _mode_question(self, question: str, mode: str, modes: List[str]) -> str:
        if mode != "sql" or len(modes) == 1:
            return question

        lowered = question.lower()
        sql_question = question.strip()

        split_patterns = [
            r",?\s+and what policy guidance.*$",
            r",?\s+and what should happen next.*$",
            r",?\s+and what policy says.*$",
            r",?\s+and what policy applies.*$",
            r",?\s+and the required policy action.*$",
            r",?\s+and required policy action.*$",
            r",?\s+and what guidance applies.*$",
        ]

        for pattern in split_patterns:
            candidate = re.sub(pattern, "", sql_question, flags=re.IGNORECASE).strip(" ,?")
            if candidate and candidate != sql_question:
                sql_question = candidate
                break

        if "highest balance" in lowered:
            return "Which account has the highest balance?"

        if "urgent according to policy" in lowered:
            return re.sub(r"\s+according to policy", "", sql_question, flags=re.IGNORECASE).strip(" ,?")

        return sql_question

    def _run_selected_modes(self, question: str, modes: List[str]) -> Dict[str, Any]:
        outputs = {}
        errors = []

        if "rag" in modes:
            try:
                outputs["rag"] = self._run_rag(question)
            except Exception as exc:
                errors.append({"mode": "rag", "error": str(exc)})

        if "sql" in modes:
            try:
                outputs["sql"] = self.sql_service.run(self._mode_question(question, "sql", modes))
            except Exception as exc:
                errors.append({"mode": "sql", "error": str(exc)})

        if "graph" in modes:
            try:
                outputs["graph"] = self.graph_service_factory().run(question)
            except Exception as exc:
                errors.append({"mode": "graph", "error": str(exc)})

        return {
            "outputs": _json_safe(outputs),
            "partial_errors": errors,
        }

    def _single_mode_answer(self, outputs: Dict[str, Any]) -> str:
        if "rag" in outputs:
            return outputs["rag"].get("answer", "")
        if "sql" in outputs:
            return outputs["sql"].get("summary", "")
        if "graph" in outputs:
            return outputs["graph"].get("summary", "")
        return ""

    def _synthesize(self, question: str, route: Dict[str, Any], results: Dict[str, Any]) -> str:
        outputs = results["outputs"]
        if len(outputs) == 1 and not results["partial_errors"]:
            return self._single_mode_answer(outputs)

        if not outputs:
            errors = "; ".join(f"{e['mode']}: {e['error']}" for e in results["partial_errors"])
            return f"AUTO could not answer because all selected pipelines failed. {errors}"

        prompt = f"""User question:
{question}

AUTO selected modes:
{json.dumps(route["modes"], ensure_ascii=False)}

Router reason:
{route["reason"]}

Pipeline outputs:
{json.dumps(outputs, ensure_ascii=False, indent=2)}

Pipeline errors:
{json.dumps(results["partial_errors"], ensure_ascii=False, indent=2)}

Write the final answer.
"""

        return self.llm.answer(
            prompt=prompt,
            system_prompt=AUTO_SYNTHESIS_SYSTEM_PROMPT,
            max_tokens=700,
        )

    def run(self, question: str) -> Dict[str, Any]:
        route = self._route(question)
        results = self._run_selected_modes(question, route["modes"])
        outputs = results["outputs"]
        answer = self._synthesize(question, route, results)

        sql_output = outputs.get("sql", {})
        graph_output = outputs.get("graph", {})
        rag_output = outputs.get("rag", {})

        return {
            "mode": "auto",
            "selected_modes": route["modes"],
            "route": {
                "confidence": route["confidence"],
                "reason": route["reason"],
                "router": route["router"],
                "requires_synthesis": route["requires_synthesis"],
            },
            "answer": answer,
            "sources": rag_output.get("sources", []),
            "sql": sql_output.get("sql", ""),
            "columns": sql_output.get("columns", []),
            "rows": sql_output.get("rows", []),
            "cypher": graph_output.get("cypher", ""),
            "results": graph_output.get("results", []),
            "pipeline_outputs": outputs,
            "partial_errors": results["partial_errors"],
        }
