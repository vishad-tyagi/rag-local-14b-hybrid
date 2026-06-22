import json
import os
import re
from typing import Any, Dict

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph

from hf_models import LangChainMLXLLM, MLXLocalLLM


NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password@4141")


CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(
    """Task: Generate a Cypher query for a Neo4j knowledge graph.

Instructions:
1. SCHEMA ADHERENCE:
   - Use only node labels, relationship types, and properties present in the schema.
   - Do not invent labels, relationship types, or properties.

2. SAFETY:
   - Generate a read-only query only.
   - Never use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, CALL, LOAD CSV, APOC, or any write operation.

3. DOCUMENT EXCLUSION:
   - Exclude raw text/source chunk nodes labeled `Document` unless the user explicitly asks for source text or documents.
   - Apply this exclusion to returned entity nodes where relevant.

4. MATCHING ENTITIES:
   - When matching entities by user text, prefer case-insensitive partial matching.
   - Use `coalesce(...)` if needed to safely match across available identifying properties.
   - Prefer the main identifier property from the schema; if unclear, use:
     `toLower(coalesce(n.id, n.name, '')) CONTAINS toLower('...')`

5. RELATIONSHIPS:
   - Prefer explicit relationship types from the schema when clearly relevant.
   - If the exact relationship is unclear, use a generic undirected relationship pattern like `-[r]-` instead of guessing.
   - The graph uses these EXACT directed relationships. Match the direction shown:
     * (Customer)-[:OWNS]->(Account)
     * (Customer)-[:ONBOARDED_AT]->(Branch)
     * (Customer)-[:HAS_RISK_SEGMENT]->(RiskSegment)
     * (Account)-[:PAID_TO]->(Merchant)
     * (Alert)-[:TRIGGERED_BY]->(Account)
     * (Alert)-[:APPLIES_TO]->(Customer)
     * (Alert)-[:REQUIRES]->(Policy)  and  (RiskSegment)-[:REQUIRES]->(Policy)
     * (Merchant)-[:SUBJECT_TO]->(Policy)
   - A customer is NOT directly connected to a merchant or an alert. Traverse:
     * customer to merchant: (Customer)-[:OWNS]->(Account)-[:PAID_TO]->(Merchant)
     * customer to alert:    (Alert)-[:APPLIES_TO]->(Customer)  (alert points to customer)
   - Do not use RELATED_TO for branch, account, merchant, alert, or policy questions unless the schema shows no more specific relationship.

5a. PROPERTIES:
   - Every node has ONLY the property `id` (for example 'Asha Rao', 'ACCT-9001', 'High Risk', 'ALERT-5001').
   - Do NOT reference any other property such as `.name`, `.score`, `.status`, `.balance`, or `.amount`. They do not exist on graph nodes.
   - Match nodes by their `id` text only, using `toLower(n.id) CONTAINS toLower('...')`.

5b. LITERAL VALUES:
   - Match the entity text exactly as a human would write it. Node ids use spaces, not hyphens: use 'high risk' NOT 'high-risk', 'low risk' NOT 'low-risk'.
   - Do not invent compound literals that are not real node ids. There is no node whose id contains 'urgent open', 'open alert', or 'high score'.
   - Concepts like "open", "urgent", "high score", or "at or above 80" are NOT stored in the graph. Do not try to filter on them in Cypher; ignore those qualifiers and return the structural relationships instead.

6. QUERY SIMPLICITY:
   - Prefer the simplest valid query that answers the question.
   - Avoid unnecessary hops, extra matches, or overly broad traversals.
   - Use variable-length paths only when needed.

7. BOOLEAN LOGIC:
   - If OR is used with AND in the same WHERE clause, wrap the OR conditions in parentheses.

8. OUTPUT:
   - Return ONLY the raw Cypher query.
   - No markdown, no explanations, no backticks, no prefix text.

Choose the best pattern based on the question. The current sample graph represents banking risk relationships such as customers, accounts, branches, merchants, alerts, risk segments, and policies.

PATTERN A - Single Entity Discovery
Example question: "What is connected to Asha Rao?"
Example query:
MATCH (n)
WHERE toLower(coalesce(n.id, n.name, '')) CONTAINS toLower('asha rao')
  AND NOT n:Document
OPTIONAL MATCH (n)-[r]-(m)
WHERE NOT m:Document
RETURN n, r, m
LIMIT 25

PATTERN B - Category Listing
Example question: "List all Customers and Merchants"
Example query:
MATCH (n)
WHERE (n:Customer OR n:Merchant) AND NOT n:Document
RETURN n
LIMIT 25

PATTERN C - Two-Entity Connection
Example question: "How is Asha Rao related to Northstar Crypto Exchange?"
Example query:
MATCH p = (a)-[*1..4]-(b)
WHERE toLower(coalesce(a.id, a.name, '')) CONTAINS toLower('asha rao')
  AND toLower(coalesce(b.id, b.name, '')) CONTAINS toLower('northstar crypto exchange')
  AND NOT a:Document
  AND NOT b:Document
RETURN p
LIMIT 5

PATTERN D - Count / Aggregation
Example question: "How many Customers are there?"
Example query:
MATCH (n:Customer)
WHERE NOT n:Document
RETURN count(n) AS count

PATTERN E - Graph Summarization / Main Concepts
Example question: "Summarize the graph" or "What are the main entities?"
Example query:
MATCH (n)-[r]-()
WHERE NOT n:Document
RETURN coalesce(n.id, n.name, 'Unknown') AS concept, count(r) AS connections
ORDER BY connections DESC
LIMIT 10

PATTERN F - Multi-Entity Connection
Example question: "Show the relationship between Asha Rao, ACCT-9001, and Northstar Crypto Exchange."
Example query:
MATCH p = (a)-[*1..4]-(c)
WHERE toLower(coalesce(a.id, a.name, '')) CONTAINS toLower('asha rao')
  AND toLower(coalesce(c.id, c.name, '')) CONTAINS toLower('northstar crypto exchange')
  AND NOT a:Document
  AND NOT c:Document
  AND any(node IN nodes(p) WHERE toLower(coalesce(node.id, node.name, '')) CONTAINS toLower('acct-9001'))
RETURN p
LIMIT 8

PATTERN G - Branch Relationship Lookup
Example question: "Which customers are connected to Downtown Branch?"
Example query:
MATCH (customer:Customer)-[r:ONBOARDED_AT]->(branch:Branch)
WHERE toLower(coalesce(branch.id, branch.name, '')) CONTAINS toLower('downtown branch')
  AND NOT customer:Document
  AND NOT branch:Document
RETURN customer, r, branch
LIMIT 25

PATTERN H - Relationship Target Lookup
Example question: "Which alerts require Urgent AML Review?"
Example query:
MATCH (source)-[r:REQUIRES]->(target)
WHERE NOT source:Document
  AND NOT target:Document
  AND toLower(coalesce(target.id, target.name, '')) CONTAINS toLower('urgent aml review')
RETURN source, r, target
LIMIT 25

Example question: "Which accounts paid to GlobeRemit Services?"
Example query:
MATCH (source)-[r:PAID_TO]->(target)
WHERE toLower(coalesce(target.id, target.name, '')) CONTAINS toLower('globeremit services')
  AND NOT source:Document
  AND NOT target:Document
RETURN source, r, target
LIMIT 25

PATTERN I - Risk-Segment Customers To Their Merchants (multi-hop)
Example question: "Which high-risk customers are connected to merchants, and which merchants are they?"
Note: ignore non-graph qualifiers like "open alerts", "urgent", or "high score"; return the structural customer-to-merchant connections.
Example query:
MATCH (customer:Customer)-[:HAS_RISK_SEGMENT]->(riskSegment:RiskSegment)
WHERE toLower(riskSegment.id) CONTAINS toLower('high risk')
MATCH (customer)-[:OWNS]->(account:Account)-[:PAID_TO]->(merchant:Merchant)
WHERE NOT customer:Document AND NOT merchant:Document
RETURN customer.id AS customer, merchant.id AS merchant
LIMIT 25

PATTERN J - Risk-Segment Customers With Their Merchant And Required Policy
Example question: "For high risk customers, give the merchant relationship and the required policy."
Note: alerts point to customers via APPLIES_TO; policies attach to the risk segment via REQUIRES.
Example query:
MATCH (customer:Customer)-[:HAS_RISK_SEGMENT]->(riskSegment:RiskSegment)
WHERE toLower(riskSegment.id) CONTAINS toLower('high risk')
MATCH (customer)-[:OWNS]->(account:Account)-[:PAID_TO]->(merchant:Merchant)
OPTIONAL MATCH (riskSegment)-[:REQUIRES]->(policy:Policy)
WHERE NOT customer:Document AND NOT merchant:Document
RETURN customer.id AS customer, merchant.id AS merchant, policy.id AS requiredPolicy
LIMIT 25

Schema:
{schema}

Question:
{question}

Cypher Query:"""
)


GRAPH_SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are a strict graph data analyst.

You MUST answer the user's question using ONLY the provided graph results.
Do not use outside knowledge. Do not guess. Do not mention Cypher, JSON, or internal tooling.

User question:
{question}

Graph results:
{results}

Rules:
1. If the graph results are empty, answer exactly: I don't have enough information based on the current graph data to answer that.
2. If the results contain a count, state the count directly.
3. If the results list entities, summarize only the returned entities and relationships.
4. If the results look like a graph overview with concepts and connections, explain that these are the most connected concepts in the current graph.
5. Be concise and grounded.

Answer:"""
)


def _strip_markdown_and_prefixes(text: str) -> str:
    cleaned = re.sub(r"<\|.*?\|>", "", text.strip())
    cleaned = cleaned.replace("```cypher", "").replace("```", "").strip()
    cleaned = re.sub(r"^Cypher Query:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Query:\s*", "", cleaned, flags=re.IGNORECASE)

    match = re.search(r"(?is)\bmatch\b.*", cleaned)
    if match:
        return match.group(0).strip()

    match = re.search(r"(?is)\breturn\b.*", cleaned)
    if match:
        return match.group(0).strip()

    return cleaned.strip()


def _ensure_read_only_cypher(cypher: str) -> str:
    normalized = cypher.strip().lower()
    if not normalized:
        raise ValueError("LLM did not generate a Cypher query.")

    blocked_patterns = [
        " create ",
        " merge ",
        " delete ",
        " detach ",
        " set ",
        " remove ",
        " drop ",
        " call ",
        " load csv ",
        " apoc.",
    ]
    padded = f" {normalized} "
    if any(token in padded for token in blocked_patterns):
        raise ValueError("Only read-only Cypher is allowed.")

    if not ("match" in normalized or normalized.startswith("return")):
        raise ValueError("LLM failed to generate a valid read-only Cypher query.")

    return cypher.strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


class GraphQueryService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
        )
        self.cypher_llm = LangChainMLXLLM()
        self.summary_llm = MLXLocalLLM()

    def _get_schema(self) -> str:
        if hasattr(self.graph, "refresh_schema"):
            self.graph.refresh_schema()
        return str(self.graph.get_schema)

    def _generate_cypher(self, question: str) -> str:
        schema = self._get_schema()
        prompt = CYPHER_GENERATION_PROMPT.format(schema=schema, question=question)
        response = self.cypher_llm.invoke(prompt)
        clean_cypher = _strip_markdown_and_prefixes(response)
        return _ensure_read_only_cypher(clean_cypher)

    def _summarize_results(self, question: str, results: Any) -> str:
        if not results:
            return "I don't have enough information based on the current graph data to answer that."

        prompt = GRAPH_SUMMARY_PROMPT.format(
            question=question,
            results=json.dumps(results, ensure_ascii=False, default=str, indent=2),
        )
        return self.summary_llm.answer(prompt, max_tokens=220).strip()

    def _fallback_cypher(self, question: str) -> str | None:
        q = question.lower()

        # Risk-segment customers -> their merchants (multi-hop). Covers AUTO
        # questions like "which high-risk customers ... connected to merchants".
        # Non-graph qualifiers (open alerts, urgent, high score) are ignored on
        # purpose; SQL handles those, the graph only holds structural links.
        risk_segments = {
            "high risk": "high risk",
            "high-risk": "high risk",
            "medium risk": "medium risk",
            "medium-risk": "medium risk",
            "low risk": "low risk",
            "low-risk": "low risk",
        }
        matched_segment = next((name for key, name in risk_segments.items() if key in q), None)
        if matched_segment and _contains_any(q, ["merchant", "merchants"]):
            if _contains_any(q, ["policy", "policies", "required", "requires"]):
                return f"""
MATCH (customer:Customer)-[:HAS_RISK_SEGMENT]->(riskSegment:RiskSegment)
WHERE toLower(riskSegment.id) CONTAINS toLower('{matched_segment}')
MATCH (customer)-[:OWNS]->(account:Account)-[:PAID_TO]->(merchant:Merchant)
OPTIONAL MATCH (riskSegment)-[:REQUIRES]->(policy:Policy)
WHERE NOT customer:Document AND NOT merchant:Document
RETURN customer.id AS customer, merchant.id AS merchant, policy.id AS requiredPolicy
LIMIT 25
""".strip()

            return f"""
MATCH (customer:Customer)-[:HAS_RISK_SEGMENT]->(riskSegment:RiskSegment)
WHERE toLower(riskSegment.id) CONTAINS toLower('{matched_segment}')
MATCH (customer)-[:OWNS]->(account:Account)-[:PAID_TO]->(merchant:Merchant)
WHERE NOT customer:Document AND NOT merchant:Document
RETURN customer.id AS customer, merchant.id AS merchant
LIMIT 25
""".strip()

        if "urgent aml review" in q and _contains_any(q, ["alert", "alerts", "require", "requires"]):
            return """
MATCH (alert:Alert)-[r:REQUIRES]->(policy:Policy)
WHERE toLower(coalesce(policy.id, policy.name, '')) CONTAINS toLower('urgent aml review')
  AND NOT alert:Document
  AND NOT policy:Document
RETURN alert, r, policy
LIMIT 25
""".strip()

        branch_names = {
            "downtown": "downtown branch",
            "lakeside": "lakeside branch",
            "airport": "airport branch",
            "capital market": "capital market branch",
        }
        for branch_key, branch_name in branch_names.items():
            if branch_key in q and _contains_any(q, ["customer", "customers", "connected", "onboarded", "branch"]):
                return f"""
MATCH (customer:Customer)-[r:ONBOARDED_AT]->(branch:Branch)
WHERE toLower(coalesce(branch.id, branch.name, '')) CONTAINS toLower('{branch_name}')
  AND NOT customer:Document
  AND NOT branch:Document
RETURN customer, r, branch
LIMIT 25
""".strip()

        merchant_names = {
            "northstar": "northstar crypto exchange",
            "globeremit": "globeremit services",
            "city utilities": "city utilities",
            "metro payroll": "metro payroll services",
            "airport atm": "airport atm network",
        }
        for merchant_key, merchant_name in merchant_names.items():
            if merchant_key in q and (
                "asha" in q
                or _contains_any(q, ["account", "accounts", "paid", "merchant", "connected", "relationship", "path"])
            ):
                if "asha" in q:
                    return f"""
MATCH p = (customer:Customer)-[*1..4]-(merchant:Merchant)
WHERE toLower(coalesce(customer.id, customer.name, '')) CONTAINS toLower('asha rao')
  AND toLower(coalesce(merchant.id, merchant.name, '')) CONTAINS toLower('{merchant_name}')
  AND NOT customer:Document
  AND NOT merchant:Document
RETURN p
LIMIT 8
""".strip()

                return f"""
MATCH (account:Account)-[r:PAID_TO]->(merchant:Merchant)
WHERE toLower(coalesce(merchant.id, merchant.name, '')) CONTAINS toLower('{merchant_name}')
  AND NOT account:Document
  AND NOT merchant:Document
RETURN account, r, merchant
LIMIT 25
""".strip()

        account_match = re.search(r"\bacct-\d+\b", q)
        if account_match and _contains_any(q, ["relationship", "connected", "path"]):
            account_id = account_match.group(0)
            return f"""
MATCH p = (left)-[*1..4]-(right)
WHERE any(node IN nodes(p) WHERE toLower(coalesce(node.id, node.name, '')) CONTAINS toLower('{account_id}'))
  AND NOT left:Document
  AND NOT right:Document
RETURN p
LIMIT 8
""".strip()

        return None

    @staticmethod
    def _has_null_cells(results: Any) -> bool:
        """A result row with null cells means the LLM query matched a partial
        path (typically a stray OPTIONAL MATCH on a non-existent value), so it
        dropped identifying columns like the customer or policy."""
        if not isinstance(results, list):
            return False
        return any(
            value is None
            for row in results
            if isinstance(row, dict)
            for value in row.values()
        )

    def run(self, question: str) -> Dict[str, Any]:
        cypher_query = self._generate_cypher(question)
        print(f"[DEBUG] Generated Cypher: {cypher_query}")

        try:
            results = self.graph.query(cypher_query)
            fallback_cypher = self._fallback_cypher(question)
            # Use the deterministic fallback when the LLM query returned nothing
            # OR returned degenerate rows (null cells). The fallback only exists
            # for recognized banking shapes and returns complete, verified rows.
            needs_fallback = not results or self._has_null_cells(results)
            if needs_fallback and fallback_cypher and fallback_cypher != cypher_query:
                fallback_results = self.graph.query(fallback_cypher)
                if fallback_results and not self._has_null_cells(fallback_results):
                    cypher_query = fallback_cypher
                    results = fallback_results

            summary = self._summarize_results(question, results)
            return {
                "cypher": cypher_query,
                "results": results,
                "summary": summary,
            }
        except Exception as e:
            return {
                "cypher": cypher_query,
                "results": [],
                "summary": f"Error executing graph query: {str(e)}",
            }
