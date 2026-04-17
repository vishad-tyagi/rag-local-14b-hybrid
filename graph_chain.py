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

6. QUERY SIMPLICITY:
   - Prefer the simplest valid query that answers the question.
   - Avoid unnecessary hops, extra matches, or overly broad traversals.
   - Use variable-length paths only when needed.

7. BOOLEAN LOGIC:
   - If OR is used with AND in the same WHERE clause, wrap the OR conditions in parentheses.

8. OUTPUT:
   - Return ONLY the raw Cypher query.
   - No markdown, no explanations, no backticks, no prefix text.

Choose the best pattern based on the question:

PATTERN A - Single Entity Discovery
Example question: "What is connected to FAISS?"
Example query:
MATCH (n)
WHERE toLower(coalesce(n.id, n.name, '')) CONTAINS toLower('faiss')
  AND NOT n:Document
OPTIONAL MATCH (n)-[r]-(m)
WHERE NOT m:Document
RETURN n, r, m
LIMIT 25

PATTERN B - Category Listing
Example question: "List all Metrics and Models"
Example query:
MATCH (n)
WHERE (n:Metric OR n:Model) AND NOT n:Document
RETURN n
LIMIT 25

PATTERN C - Two-Entity Connection
Example question: "How is X related to Y?"
Example query:
MATCH p = (a)-[*1..2]-(b)
WHERE toLower(coalesce(a.id, a.name, '')) CONTAINS toLower('x')
  AND toLower(coalesce(b.id, b.name, '')) CONTAINS toLower('y')
  AND NOT a:Document
  AND NOT b:Document
RETURN p
LIMIT 5

PATTERN D - Count / Aggregation
Example question: "How many Metrics are there?"
Example query:
MATCH (n:Metric)
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
Example question: "How is X related to Y and Z?"
Example query:
MATCH p = (a)-[*1..2]-(b)
WHERE toLower(coalesce(a.id, a.name, '')) CONTAINS toLower('x')
  AND (
    toLower(coalesce(b.id, b.name, '')) CONTAINS toLower('y')
    OR toLower(coalesce(b.id, b.name, '')) CONTAINS toLower('z')
  )
  AND NOT a:Document
  AND NOT b:Document
RETURN p
LIMIT 8

PATTERN G - Relationship Target Lookup
Example question: "What concepts optimize retrieval accuracy or faithfulness?"
Example query:
MATCH (c:Concept)-[r:OPTIMIZES]->(target)
WHERE NOT c:Document
  AND NOT target:Document
  AND (
    toLower(coalesce(target.id, target.name, '')) CONTAINS toLower('retrieval accuracy')
    OR toLower(coalesce(target.id, target.name, '')) CONTAINS toLower('faithfulness')
  )
RETURN c, r, target
LIMIT 25

Example question: "What does RAGAS evaluate?"
Example query:
MATCH (source)-[r:EVALUATES]->(target)
WHERE toLower(coalesce(source.id, source.name, '')) CONTAINS toLower('ragas')
  AND NOT source:Document
  AND NOT target:Document
RETURN source, r, target
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

    def run(self, question: str) -> Dict[str, Any]:
        cypher_query = self._generate_cypher(question)
        print(f"[DEBUG] Generated Cypher: {cypher_query}")

        try:
            results = self.graph.query(cypher_query)
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
