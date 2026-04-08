# import os
# from typing import Any, Dict, List
# from langchain_neo4j import Neo4jGraph
# from langchain_core.prompts import PromptTemplate
# from hf_models import LangChainMLXLLM, MLXLocalLLM

# # Configuration
# NEO4J_URL = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PASSWORD = "password@4141"  # Change to your Neo4j password

# # --- ENHANCEMENT 1: ADDED PATTERN D FOR GRAPH ALGORITHMS ---
# CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(
#     """Task: Generate a Cypher query for a Neo4j knowledge graph.

# Instructions:
# 1. SCHEMA ADHERENCE:
#    - Use only node labels, relationship types, and properties present in the schema.
#    - Do not invent labels, relationship types, or properties.

# 2. DOCUMENT EXCLUSION:
#    - Exclude raw text/source chunk nodes labeled `Document` unless the user explicitly asks for source text or documents.
#    - Apply this exclusion to all returned entity nodes where relevant.

# 3. MATCHING ENTITIES:
#    - When matching entities by user text, prefer case-insensitive partial matching.
#    - Use `coalesce(...)` if needed to safely match across available identifying properties.
#    - Prefer the main identifier property from the schema; if unclear, use a safe fallback like:
#      `toLower(coalesce(n.id, n.name, '')) CONTAINS toLower('...')`

# 4. RELATIONSHIPS:
#    - Prefer explicit relationship types from the schema when clearly relevant.
#    - If the exact relationship is unclear, use a generic undirected relationship pattern like `-[r]-` instead of guessing.

# 5. QUERY SIMPLICITY:
#    - Prefer the simplest valid query that answers the question.
#    - Avoid unnecessary hops, extra matches, or overly broad traversals.
#    - Use variable-length paths only when needed.

# 6. OUTPUT:
#    - Return ONLY the raw Cypher query.
#    - No markdown, no explanations, no backticks.

# 7. BOOLEAN LOGIC (AND/OR):
#    - If you use `OR` alongside `AND` in a `WHERE` clause, you MUST wrap the `OR` conditions in parentheses to prevent precedence errors.
#    - Example: `WHERE a.id CONTAINS 'x' AND (b.id CONTAINS 'y' OR b.id CONTAINS 'z') AND NOT a:Document`

# Choose the best pattern based on the question:

# PATTERN A - Single Entity Discovery
# Example question: "What is connected to FAISS?"
# Example query:
# MATCH (n)
# WHERE toLower(coalesce(n.id, n.name, '')) CONTAINS toLower('faiss')
#   AND NOT n:Document
# OPTIONAL MATCH (n)-[r]-(m)
# WHERE NOT m:Document
# RETURN n, r, m
# LIMIT 25

# PATTERN B - Category Listing
# Example question: "List all Metrics and Algorithms"
# Example query:
# MATCH (n)
# WHERE (n:Metric OR n:Algorithm) AND NOT n:Document
# RETURN n
# LIMIT 25

# PATTERN C - Two-Entity Connection
# Example question: "How is X related to Y?"
# Example query:
# MATCH p = (a)-[*1..2]-(b)
# WHERE toLower(coalesce(a.id, a.name, '')) CONTAINS toLower('x')
#   AND toLower(coalesce(b.id, b.name, '')) CONTAINS toLower('y')
#   AND NOT a:Document
#   AND NOT b:Document
# RETURN p
# LIMIT 5

# PATTERN D - Count / Aggregation
# Example question: "How many Metrics are there?"
# Example query:
# MATCH (n:Metric)
# WHERE NOT n:Document
# RETURN count(n) AS count

# PATTERN E - Graph Summarization & Importance
# Example question: "Summarize the graph" or "What are the core concepts?"
# Example query:
# MATCH (n)-[r]-() 
# WHERE NOT n:Document 
# RETURN coalesce(n.id, n.name, 'Unknown') AS Concept, count(r) AS Connections 
# ORDER BY Connections DESC 
# LIMIT 5

# PATTERN F - Multi-Entity Connection (3 or more concepts)
# Example question: "How is X related to Y and Z?"
# Example query:
# MATCH p = (a)-[*1..2]-(b)
# WHERE toLower(coalesce(a.id, a.name, '')) CONTAINS toLower('x')
#   AND (toLower(coalesce(b.id, b.name, '')) CONTAINS toLower('y') OR toLower(coalesce(b.id, b.name, '')) CONTAINS toLower('z'))
#   AND NOT a:Document
#   AND NOT b:Document
# RETURN p
# LIMIT 5

# Schema:
# {schema}

# Question:
# {question}

# Cypher Query:"""
# )

# class GraphQueryService:
#     def __init__(self):
#         self.graph = Neo4jGraph(
#             url=NEO4J_URL, 
#             username=NEO4J_USER, 
#             password=NEO4J_PASSWORD
#         )
#         self.cypher_llm = LangChainMLXLLM()
#         self.summary_llm = MLXLocalLLM()

#     def _get_schema(self) -> str:
#         return self.graph.get_schema

#     def _generate_cypher(self, question: str) -> str:
#         schema = self._get_schema()
#         prompt = CYPHER_GENERATION_PROMPT.format(schema=schema, question=question)
#         response = self.cypher_llm.invoke(prompt)
        
#         # 1. Strip markdown blocks
#         clean_cypher = response.replace("```cypher", "").replace("```", "").strip()
        
#         # 2. NEW: Forcefully remove the "Cypher Query:" prefix if the 14B model outputs it
#         if clean_cypher.lower().startswith("cypher query:"):
#             clean_cypher = clean_cypher[len("cypher query:"):].strip()
            
#         return clean_cypher

#     def run(self, question: str) -> Dict[str, Any]:
#         cypher_query = self._generate_cypher(question)
        
#         try:
#             results = self.graph.query(cypher_query)
            
#             # --- ENHANCEMENT 2: STRICT GROUNDING ENFORCEMENT ---
#             summary_prompt = f"""
#             You are a strict data analyst. You MUST answer the user's question based ONLY on the provided Graph Context.
            
#             User Question: "{question}"
            
#             --- Graph Context ---
#             {results if results else "[]"}
            
#             Instructions:
#             1. THE GOLDEN RULE: If the Graph Context is empty ("[]"), you MUST output EXACTLY: "I don't have enough information based on the current data to answer that." DO NOT guess. DO NOT use external knowledge.
#             2. If the Graph Context contains "Concept" and "Connections" (Summary Mode), list the top concepts and explain that they are the most central topics based on their connections in the graph.
#             3. If the Graph Context contains relationships (n, r, m), synthesize a natural answer explaining how the entities are connected.
#             4. Do not mention "Cypher", "JSON", or "Graph Context" in your response. Just give the answer.
#             """
            
#             summary = self.summary_llm.answer(summary_prompt, max_tokens=300)
            
#             return {
#                 "cypher": cypher_query,
#                 "results": results,
#                 "summary": summary
#             }
#         except Exception as e:
#             return {
#                 "cypher": cypher_query,
#                 "results": [],
#                 "summary": f"Error executing graph query: {str(e)}"
#             }













# // GPT prompt in below code



import os
from typing import Any, Dict, List
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from hf_models import LangChainMLXLLM, MLXLocalLLM

# Configuration
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password@4141" # Use the password you set in Neo4j Desktop


CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(
    """Task: Generate a Cypher query for a Neo4j knowledge graph.

Instructions:
1. SCHEMA ADHERENCE:
   - Use only node labels, relationship types, and properties present in the schema.
   - Do not invent labels, relationship types, or properties.

2. DOCUMENT EXCLUSION:
   - Exclude raw text/source chunk nodes labeled `Document` unless the user explicitly asks for source text or documents.
   - Apply this exclusion to all returned entity nodes where relevant.

3. MATCHING ENTITIES:
   - When matching entities by user text, prefer case-insensitive partial matching.
   - Use `coalesce(...)` if needed to safely match across available identifying properties.
   - Prefer the main identifier property from the schema; if unclear, use a safe fallback like:
     `toLower(coalesce(n.id, n.name, '')) CONTAINS toLower('...')`

4. RELATIONSHIPS:
   - Prefer explicit relationship types from the schema when clearly relevant.
   - If the exact relationship is unclear, use a generic undirected relationship pattern like `-[r]-` instead of guessing.

5. QUERY SIMPLICITY:
   - Prefer the simplest valid query that answers the question.
   - Avoid unnecessary hops, extra matches, or overly broad traversals.
   - Use variable-length paths only when needed.

6. OUTPUT:
   - Return ONLY the raw Cypher query.
   - No markdown, no explanations, no backticks.

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
Example question: "List all Metrics"
Example query:
MATCH (n:Metric)
WHERE NOT n:Document
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

Schema:
{schema}

Question:
{question}

Cypher Query:"""
)


# CYPHER_GENERATION_PROMPT = PromptTemplate.from_template(
#     """Task: Generate a Cypher statement to query a specialized knowledge graph.

# Instructions:
# 1. SCHEMA ADHERENCE: Use only the labels and relationships provided in the schema.
# 2. EXCLUSION: Always exclude raw text chunks using `WHERE NOT n:Document` and `WHERE NOT m:Document`.
# 3. FLEXIBILITY: Do not guess relationship names unless explicitly asked. Use `-[r]-` for general connections.
# 4. CHOOSE THE CORRECT PATTERN based on the question type:

#    PATTERN A - Single Entity Discovery (e.g., "What is connected to FAISS?"):
#    MATCH (n) WHERE toLower(n.id) CONTAINS toLower('faiss') AND NOT n:Document 
#    OPTIONAL MATCH (n)-[r]-(m) WHERE NOT m:Document 
#    RETURN n, r, m LIMIT 25

#    PATTERN B - Category Listing (e.g., "List all Metrics"):
#    MATCH (n:Metric) WHERE NOT n:Document RETURN n LIMIT 25

#    PATTERN C - Two-Entity Connection (e.g., "How is X related to Y?"):
#    MATCH p = (a)-[*1..3]-(b) WHERE toLower(a.id) CONTAINS toLower('x') AND toLower(b.id) CONTAINS toLower('y') AND NOT a:Document AND NOT b:Document RETURN p LIMIT 5

# Schema:
# {schema}

# Question:
# {question}

# Return ONLY the raw Cypher query. No markdown formatting, no explanations.
# Cypher Query:"""
# )



class GraphQueryService:
    def __init__(self):
        # 1. Connect to Neo4j
        self.graph = Neo4jGraph(
            url=NEO4J_URL, 
            username=NEO4J_USER, 
            password=NEO4J_PASSWORD
        )
        
        # 2. Initialize LLMs (Using your existing MLX wrappers)
        self.cypher_llm = LangChainMLXLLM()
        self.summary_llm = MLXLocalLLM()

    def _get_schema(self) -> str:
        return self.graph.get_schema

    def _generate_cypher(self, question: str) -> str:
        schema = self._get_schema()
        prompt = CYPHER_GENERATION_PROMPT.format(schema=schema, question=question)
        
        # Generate raw output
        response = self.cypher_llm.invoke(prompt)
        
        # Clean up the response (remove markdown code blocks if present)
        clean_cypher = response.replace("```cypher", "").replace("```", "").strip()
        return clean_cypher

    def run(self, question: str) -> Dict[str, Any]:
        # Step 1: Generate Cypher
        cypher_query = self._generate_cypher(question)
        print(f"[DEBUG] Generated Cypher: {cypher_query}")
        
        try:
            # Step 2: Execute Query
            results = self.graph.query(cypher_query)
            
            # Step 3: Summarize Results
            summary_prompt = f"""
            User Question: {question}
            Graph Query Results: {results}
            
            Provide a concise and helpful summary of these results based on the graph data.
            """
            summary = self.summary_llm.answer(summary_prompt, max_tokens=300)
            
            return {
                "cypher": cypher_query,
                "results": results,
                "summary": summary
            }
        except Exception as e:
            return {
                "cypher": cypher_query,
                "results": [],
                "summary": f"Error executing graph query: {str(e)}"
            }