"""
src/agents/sql_agent.py
────────────────────────
Agent 2a — SQL Agent.

Converts natural language procurement questions into SQL,
executes against PostgreSQL, and returns structured + narrated results.

Pipeline:
  1. Build prompt with schema context + question
  2. Claude Haiku generates a safe SELECT query
  3. SQLAlchemy executes query against PostgreSQL
  4. Claude Haiku narrates the raw results in plain English
  5. Returns SQLAgentResult with query, raw data, and narrative

Safety:
  - Only SELECT statements permitted (write operations blocked)
  - Query length capped to prevent runaway generation
  - Results capped at 50 rows

Usage:
    from src.agents.sql_agent import SQLAgent
    agent  = SQLAgent()
    result = agent.run("What is our total open PO value for IT Services this quarter?")
    print(result.narrative)
    print(result.data)
"""

import re
import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text
from loguru import logger

from src.agents.bedrock_client import get_llm
from src.data.db_loader import get_engine, get_session, SCHEMA_DESCRIPTION


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class SQLAgentResult:
    question   : str
    sql_query  : str               # the generated SQL
    data       : list[dict]        # raw rows as list of dicts
    narrative  : str               # plain-English answer
    row_count  : int = 0
    error      : str = ""
    success    : bool = True

    def to_dict(self) -> dict:
        return {
            "question" : self.question,
            "sql_query": self.sql_query,
            "data"     : self.data,
            "narrative": self.narrative,
            "row_count": self.row_count,
            "error"    : self.error,
            "success"  : self.success,
        }


# ── Prompts ───────────────────────────────────────────────────────────────────

SQL_GENERATION_SYSTEM = f"""You are a PostgreSQL expert for a procurement ERP system.

{SCHEMA_DESCRIPTION}

RULES:
1. Generate ONLY a single SELECT statement — no INSERT, UPDATE, DELETE, DROP, or DDL.
2. Always use table aliases (v for vendors, po for purchase_orders, inv for invoices, sa for spend_analysis).
3. Cap results at 50 rows using LIMIT 50 unless user asks for all.
4. For "this quarter": use EXTRACT(QUARTER FROM po.po_date) = EXTRACT(QUARTER FROM CURRENT_DATE)
   AND EXTRACT(YEAR FROM po.po_date) = EXTRACT(YEAR FROM CURRENT_DATE)
5. For "overdue invoices": status = 'overdue' OR (status != 'paid' AND due_date < CURRENT_DATE)
6. Always include vendor name in results (JOIN vendors) when querying purchase_orders or invoices.
7. Format money columns with ROUND(expression::numeric, 2) — always cast to numeric first.
   Example: ROUND(AVG(po.amount)::numeric, 2), ROUND(SUM(po.amount)::numeric, 2)
   PostgreSQL's ROUND(x, n) requires numeric type; float/double precision must be cast explicitly.
8. Return ONLY the raw SQL query — no explanation, no markdown, no semicolons at the end.
9. For spending/summary questions, use simple aggregation on vendors + purchase_orders only.
   Do NOT join spend_analysis with purchase_orders — use spend_analysis standalone for trend data.
   Do NOT add quarter/date filters unless the question explicitly asks for a time period.

Today's date: {date.today().isoformat()}"""

SQL_NARRATIVE_SYSTEM = """You are a procurement analyst presenting database results to a business user.

Given a question and its query results (as JSON), write a clear, concise 2-4 sentence answer.

Rules:
- State the key finding in the first sentence
- Include specific numbers from the data
- Use business language (not technical SQL language)
- If results are empty, say so clearly and suggest why
- Do not mention SQL or databases
- Format currency as $X,XXX format"""


# ── SQL Agent ─────────────────────────────────────────────────────────────────

class SQLAgent:
    """
    Two-step agent:
      Step 1 — Generate SQL from natural language question
      Step 2 — Narrate the results in plain English
    """

    def __init__(self):
        self._llm_sql       = get_llm(temperature=0.0, max_tokens=512)
        self._llm_narrative = get_llm(temperature=0.2, max_tokens=512)
        self._engine        = get_engine()
        logger.info("SQLAgent initialised")

    # ── Step 1: Generate SQL ──────────────────────────────────────────────────

    def _generate_sql(self, question: str) -> str:
        """Convert natural language question to a SQL SELECT statement."""
        messages = [
            SystemMessage(content=SQL_GENERATION_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nSQL query:"),
        ]
        response = self._llm_sql.invoke(messages)
        sql = response.content.strip()

        # Strip markdown code fences if present
        sql = re.sub(r"```sql|```", "", sql).strip()
        # Remove trailing semicolons
        sql = sql.rstrip(";").strip()

        logger.debug(f"  Generated SQL:\n{sql}")
        return sql

    # ── Step 2: Execute SQL ───────────────────────────────────────────────────

    def _execute_sql(self, sql: str) -> tuple[list[dict], str]:
        """
        Execute the SQL query safely.
        Returns (rows_as_dicts, error_string).
        error_string is empty on success.
        """
        # Safety check — only allow SELECT
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
            return [], "BLOCKED: Only SELECT queries are permitted."

        try:
            with self._engine.connect() as conn:
                result  = conn.execute(text(sql))
                columns = list(result.keys())
                rows    = result.fetchmany(50)   # hard cap at 50 rows

                data = []
                for row in rows:
                    row_dict = {}
                    for col, val in zip(columns, row):
                        # Serialise non-JSON-native types
                        if isinstance(val, date):
                            row_dict[col] = val.isoformat()
                        elif isinstance(val, Decimal):
                            row_dict[col] = float(val)
                        elif val is None:
                            row_dict[col] = None
                        else:
                            row_dict[col] = val
                    data.append(row_dict)

                logger.debug(f"  Query returned {len(data)} rows")
                return data, ""

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"  SQL execution error: {error_msg}")
            return [], error_msg

    # ── Step 3: Narrate results ───────────────────────────────────────────────

    def _narrate_results(self, question: str, sql: str, data: list[dict]) -> str:
        """Generate a plain-English answer from the query results."""
        if not data:
            return "No records were found matching your query in the ERP database."

        # Truncate data preview to avoid overwhelming the LLM
        preview = data[:10] if len(data) > 10 else data
        data_json = json.dumps(preview, indent=2)

        messages = [
            SystemMessage(content=SQL_NARRATIVE_SYSTEM),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"Query results ({len(data)} rows total, showing first {len(preview)}):\n"
                f"{data_json}\n\n"
                f"Write a clear business answer:"
            )),
        ]

        response = self._llm_narrative.invoke(messages)
        return response.content.strip()

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, question: str) -> SQLAgentResult:
        """
        Full SQL agent pipeline: question → SQL → execute → narrate.

        Args:
            question : Natural language procurement question.

        Returns:
            SQLAgentResult with generated SQL, raw data, and narrative.
        """
        logger.info(f"SQLAgent running: '{question[:80]}'")

        # Step 1: Generate SQL
        try:
            sql = self._generate_sql(question)
        except Exception as e:
            logger.error(f"  SQL generation failed: {e}")
            return SQLAgentResult(
                question=question, sql_query="", data=[],
                narrative=f"Unable to generate a database query for this question. Error: {e}",
                success=False, error=str(e),
            )

        # Step 2: Execute SQL
        data, error = self._execute_sql(sql)

        if error:
            # Retry once with error context.
            # FIX: prompt explicitly demands only SQL — no explanation text —
            # which previously caused a second syntax error on retry.
            logger.warning(f"  Retrying SQL after error: {error}")
            retry_messages = [
                SystemMessage(content=SQL_GENERATION_SYSTEM),
                HumanMessage(content=(
                    f"Question: {question}\n\n"
                    f"Previous attempt generated this SQL:\n{sql}\n\n"
                    f"It produced this error: {error}\n\n"
                    f"Return ONLY the corrected SQL query. "
                    f"No explanation, no comments, no text before or after the SQL."
                )),
            ]
            try:
                fixed = self._llm_sql.invoke(retry_messages).content.strip()
                # Strip markdown fences
                fixed = re.sub(r"```sql|```", "", fixed).strip()
                # FIX: extract only the SELECT/WITH block — drop any prose the
                # LLM appended after the query (e.g. "The error indicates...")
                match = re.search(
                    r"((?:WITH|SELECT)\b.*)",
                    fixed,
                    re.IGNORECASE | re.DOTALL,
                )
                if match:
                    fixed = match.group(1).strip()
                fixed = fixed.rstrip(";").strip()
                logger.debug(f"  Retry SQL:\n{fixed}")
                data, error = self._execute_sql(fixed)
                sql = fixed
            except Exception as retry_e:
                logger.error(f"  Retry failed: {retry_e}")

        if error:
            return SQLAgentResult(
                question=question, sql_query=sql, data=[],
                narrative=f"The database query encountered an error: {error}",
                success=False, error=error,
            )

        # Step 3: Narrate
        narrative = self._narrate_results(question, sql, data)

        result = SQLAgentResult(
            question  = question,
            sql_query = sql,
            data      = data,
            narrative = narrative,
            row_count = len(data),
            success   = True,
        )

        logger.success(
            f"  SQLAgent done: {len(data)} rows → narrative generated"
        )
        return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = SQLAgent()

    test_questions = [
        "What is our total open PO value for IT Services vendors this quarter?",
        "Show all invoices overdue by more than 30 days grouped by category.",
        "Who are our top 5 vendors by total PO value in 2024?",
        "How many vendors do we have per country?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = agent.run(q)
        print(f"\nSQL:\n{result.sql_query}")
        print(f"\nRows: {result.row_count}")
        print(f"\nAnswer: {result.narrative}")
