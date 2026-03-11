import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool
from langchain_community.utilities import SQLDatabase

from hf_models import LangChainMLXLLM, MLXLocalLLM

DB_PATH = Path("data.db").resolve()

SQL_SYSTEM_PROMPT = (
    "You are Butler, an expert SQLite assistant. "
    "Return ONLY valid SQLite SQL. "
    "Do not add markdown fences. "
    "Do not add explanations. "
    "Only produce read-only SQL."
)

SQL_SUMMARY_SYSTEM_PROMPT = (
    "You are Butler. Summarize SQL query results clearly and concisely for the user."
)

SQL_PROMPT = PromptTemplate.from_template(
    """You are an expert SQLite assistant.

Your only job is to generate ONE valid SQLite query.

Rules:
- Return ONLY raw SQLite SQL
- Return exactly ONE SQL statement
- Do NOT include markdown fences
- Do NOT include explanation
- Do NOT include notes
- Do NOT include prose such as "Here is the SQL" or "Let me"
- Do NOT answer in natural language
- The query must begin with SELECT or WITH
- Only generate read-only SQL
- Use only tables and columns that exist in the schema
- If the user asks for a specific item, use a WHERE clause
- If the user does NOT specify how many rows they want, limit results to at most {top_k} rows
- If the user explicitly asks for a specific number of rows, use that number in LIMIT
- If the user explicitly asks for ALL rows, do NOT include LIMIT
- For counting questions, use COUNT(...)
- For stock questions, use the stock_quantity column

Schema:
{table_info}

Question: {input}

Return ONLY SQL:
"""
)

# SQL_PROMPT = PromptTemplate.from_template(
#     """You are an expert SQLite assistant.

# Given the user's question and the database schema, write a syntactically correct SQLite query.

# Rules:
# - Return ONLY valid SQL
# - No markdown fences
# - No explanation
# - Only generate read-only SQL (SELECT queries only)
# - Use only tables and columns that exist in the schema
# - Prefer concise queries
# - If the user does NOT specify how many rows they want, limit results to at most {top_k} rows
# - If the user explicitly asks for a specific number of rows, use that number in a LIMIT clause
# - If the user explicitly asks for ALL rows, do NOT include a LIMIT clause

# Schema:
# {table_info}

# Question: {input}

# SQLQuery:"""
# )


def strip_markdown_sql(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```sql", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = re.sub(r"^SQLQuery:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    # Extract first SQL statement starting with SELECT or WITH
    match = re.search(r"(?is)\b(select|with)\b.*?(;|$)", cleaned)
    if match:
        cleaned = match.group(0).strip()

    return cleaned


def ensure_read_only_sql(sql: str) -> str:
    normalized = sql.strip().lower()
    blocked = [
        "insert", "update", "delete", "drop", "alter", "create",
        "attach", "detach", "replace", "truncate", "vacuum", "pragma"
    ]

    if any(word in normalized for word in blocked):
        raise ValueError("Only read-only SQL is allowed.")

    if not (normalized.startswith("select") or normalized.startswith("with")):
        raise ValueError("Only SELECT/CTE read-only queries are allowed.")

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
        # Read-only SQLAlchemy connection for LangChain schema/query tools
        uri = f"sqlite:///file:{DB_PATH.as_posix()}?mode=ro&uri=true"
        engine = create_engine(uri, connect_args={"uri": True})
        self.db = SQLDatabase(engine)

        self.sql_llm = LangChainMLXLLM()
        self.summary_llm = MLXLocalLLM()

        self.sql_chain = create_sql_query_chain(
            llm=self.sql_llm,
            db=self.db,
            prompt=SQL_PROMPT,
            k=20,
        )

        self.checker = QuerySQLCheckerTool(
            db=self.db,
            llm=self.sql_llm,
        )

    def _generate_sql(self, question: str) -> str:
        raw_sql = self.sql_chain.invoke({"question": question})
        cleaned_sql = strip_markdown_sql(raw_sql)
        checked_sql = self.checker.invoke({"query": cleaned_sql})
        checked_sql = strip_markdown_sql(checked_sql)
        checked_sql = ensure_read_only_sql(checked_sql)
        print("RAW SQL FROM LLM:", raw_sql)
        print("CLEANED SQL:", cleaned_sql)
        return checked_sql

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

        try:
            fixed_sql = self.checker.invoke({"query": fixed_sql})
        except Exception:
            pass

        fixed_sql = strip_markdown_sql(fixed_sql)
        fixed_sql = ensure_read_only_sql(fixed_sql)
        return fixed_sql

    def _summarize_result(self, question: str, sql: str, columns: List[str], rows: List[Dict[str, Any]]) -> str:
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