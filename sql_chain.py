import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase

from hf_models import LangChainMLXLLM, MLXLocalLLM

DB_PATH = Path("data.db").resolve()

SQL_SYSTEM_PROMPT = (
    "You are Butler, an expert SQLite assistant. "
    "Your only output must be raw, executable SQLite code. "
    "Do not output conversational text, markdown formatting, or explanations. "
    "Only produce read-only SQL."
)

# SQL_SUMMARY_SYSTEM_PROMPT = (
#     "You are Butler. Summarize SQL query results clearly and concisely for the user."
# )

SQL_SUMMARY_SYSTEM_PROMPT = (
    "You are Butler, an expert data analyst. "
    "Your job is to summarize the results of a SQL query in natural language. "
    "Be concise, direct, and helpful. "
    "Do NOT mention the SQL code itself or how the query was constructed; just explain the data results to the user."
)



# SQL_PROMPT = PromptTemplate.from_template(
#     """You are a strict SQLite query generator.

# Your ONLY job is to output ONE valid, read-only SQLite query. You must not output any conversational text, explanations, or formatting.

# Rules:
# - Output ONLY the raw SQL query.
# - Do NOT wrap the SQL in markdown formatting (e.g., no ```sql).
# - Do NOT output any tables, notes, or natural language prose (like "Here is the query...").
# - The query MUST begin with SELECT or WITH and end with a semicolon (;).
# - Use only tables and columns that exist in the provided schema.
# - If the user asks for a specific item, use a WHERE clause.
# - If the user asks for a specific number of rows, use that exact number in the LIMIT clause.
# - If the user explicitly asks for ALL rows, omit the LIMIT clause.
# - For counting questions, use COUNT(...).
# - For stock or inventory questions, use the stock_quantity column.
# - If the question implies a single best/worst/highest/lowest result (e.g., "the most", "the cheapest", "the top category"), you MUST use LIMIT 1.
# - If the user's question uses plural terms like "items," "products," or "which ones," use LIMIT {top_k} even for superlative questions (e.g., "most expensive items").
# - For all other queries where no specific limit or count is requested, default to LIMIT {top_k}.
# - Use the 'AS' keyword to give meaningful names to calculated columns (e.g., AS percentage)

# Schema:
# {table_info}

# Question: {input}

# SQLQuery: """
# )

SQL_PROMPT = PromptTemplate.from_template(
    """You are a strict SQLite query generator.

Your ONLY job is to output ONE valid, read-only SQLite query. You must not output any conversational text, explanations, or formatting.

Rules:
- Output ONLY the raw SQL query.
- Do NOT wrap the SQL in markdown formatting.
- The query MUST begin with SELECT or WITH and end with a semicolon (;).
- Use only tables and columns that exist in the provided schema.
- For stock or inventory questions, use the stock_quantity column.
- Use the 'AS' keyword to give meaningful names to calculated columns.
- QUANTITY LOGIC:
    1. If the question asks for "the" single best/worst/highest (singular), use LIMIT 1.
    2. If the question asks for "items", "products", or "ones" (plural), use LIMIT {top_k} even for superlative questions.
    3. If the user specifies a number (e.g., "top 5"), use that exact LIMIT.
    4. If the user asks for "ALL", omit the LIMIT.
    5. Default to LIMIT {top_k} for all other general queries.

Examples:
Question: What is the most expensive product?
SQLQuery: SELECT name, price FROM products ORDER BY price DESC LIMIT 1;

Question: Show me the most expensive products.
SQLQuery: SELECT name, price FROM products ORDER BY price DESC LIMIT {top_k};

Question: Which item has the lowest stock?
SQLQuery: SELECT name, stock_quantity FROM products ORDER BY stock_quantity ASC LIMIT 1;

Schema:
{table_info}

Question: {input}

SQLQuery: """
)

def strip_markdown_sql(text: str) -> str:
    # 1. Strip special LLM chat tokens like <|im_end|> FIRST
    cleaned = re.sub(r"<\|.*?\|>", "", text.strip())
    
    # 2. Remove LangChain prefixes and markdown formatting
    cleaned = re.sub(r"^SQLQuery:\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```sql", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    
    # 3. Slice off hallucinated markdown tables (safely, only on new lines)
    if "\n|" in cleaned:
        cleaned = cleaned.split("\n|")[0]

    # 4. Extract the actual SQL statement
    match = re.search(r"(?is)\b(select|with)\b.*?(;|$)", cleaned)
    if match:
        return match.group(0).strip()

    return cleaned.strip()


def ensure_read_only_sql(sql: str) -> str:
    normalized = sql.strip().lower()
    
    # NEW: Failsafe if the model returned something that isn't a query at all
    if not (normalized.startswith("select") or normalized.startswith("with")):
        raise ValueError("LLM failed to generate a valid SELECT or WITH statement.")

    blocked = [
        "insert", "update", "delete", "drop", "alter", "create",
        "attach", "detach", "replace", "truncate", "vacuum", "pragma"
    ]

    if any(word in normalized for word in blocked):
        raise ValueError("Only read-only SQL is allowed.")

    return sql.strip()


def execute_read_only_sql(sql: str) -> Dict[str, Any]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute(sql)

    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description] if cur.description else []

    conn.close()

    result_rows = []
    for row in rows:
        result_rows.append(dict(zip(columns, row)))

    return {
        "columns": columns,
        "rows": result_rows,
    }


class SQLQueryService:
    def __init__(self):
        # Read-only SQLAlchemy connection for LangChain schema inspection
        uri = f"sqlite:///file:{DB_PATH.as_posix()}?mode=ro&uri=true"
        engine = create_engine(uri, connect_args={"uri": True})
        self.db = SQLDatabase(engine)

        self.sql_llm = LangChainMLXLLM()
        self.summary_llm = MLXLocalLLM()

        # Default top_k = 5 when user does not ask for a larger number
        self.sql_chain = create_sql_query_chain(
            llm=self.sql_llm,
            db=self.db,
            prompt=SQL_PROMPT,
            k=20,
        )

    def _generate_sql(self, question: str) -> str:
        raw_sql = self.sql_chain.invoke({"question": question})
    
        # --- ADD THIS DEBUG LINE ---
        print(f"\n[DEBUG] LLM RAW OUTPUT:\n{raw_sql}\n[DEBUG] END RAW\n")
        # ---------------------------

        cleaned_sql = strip_markdown_sql(raw_sql)
        print(f"[DEBUG] CLEANED SQL: {cleaned_sql}")
    
        cleaned_sql = ensure_read_only_sql(cleaned_sql)
        return cleaned_sql

    def _repair_sql(self, question: str, failed_sql: str, error_msg: str) -> str:
        schema = self.db.get_table_info()

        repair_prompt = f"""The previous SQL failed.

User question:
{question}

Schema:
{schema}

Previous SQL:
{failed_sql}

The previous SQL failed with this error:
{error_msg}

Please correct the syntax and provide only the fixed SQL.
"""

        fixed_sql = self.summary_llm.answer(
            prompt=repair_prompt,
            system_prompt=SQL_SYSTEM_PROMPT,
            max_tokens=256,
        )

        fixed_sql = strip_markdown_sql(fixed_sql)
        fixed_sql = ensure_read_only_sql(fixed_sql)
        return fixed_sql

    def _summarize_result(
        self,
        question: str,
        sql: str,
        columns: List[str],
        rows: List[Dict[str, Any]],
    ) -> str:
        if not rows:
            return "No rows were returned for that query."

        preview = rows[:20]

        prompt = f"""User question:
{question}

SQL used:
{sql}

Columns:
{json.dumps(columns, ensure_ascii=False)}

Rows:
{json.dumps(preview, ensure_ascii=False, indent=2)}

Write a concise natural-language summary of the result.
"""

        return self.summary_llm.answer(
            prompt=prompt,
            system_prompt=SQL_SUMMARY_SYSTEM_PROMPT,
            max_tokens=220,
        )

    def run(self, question: str) -> Dict[str, Any]:
        sql = self._generate_sql(question)

        try:
            result = execute_read_only_sql(sql)
        except (sqlite3.Error, sqlite3.ProgrammingError, ValueError) as e:
            fixed_sql = self._repair_sql(question, sql, str(e))
            result = execute_read_only_sql(fixed_sql)
            sql = fixed_sql

        summary = self._summarize_result(
            question=question,
            sql=sql,
            columns=result["columns"],
            rows=result["rows"],
        )

        return {
            "sql": sql,
            "columns": result["columns"],
            "rows": result["rows"],
            "summary": summary,
        }