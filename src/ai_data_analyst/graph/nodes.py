"""LangGraph nodes for the text-to-SQL data analyst workflow.

Workflow:
    generate_sql -> execute_sql -> (fix_sql loop) -> generate_answer -> suggest_chart
    Any dead-end (no API key, invalid SQL, empty result, LLM failure) routes to
    a ``no_info`` node that answers "No info available".
"""
import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ai_data_analyst.config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    HAS_API_KEY,
    MAX_SQL_RETRIES,
    NO_INFO_MESSAGE,
    TOP_K,
    TOP_P,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)
from ai_data_analyst.graph.sql_engine import SqlEngine
from ai_data_analyst.graph.state import AnalystState

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore[assignment,misc]

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert data analyst that converts natural language questions into DuckDB SQL.

Database schema:
{schema}

Rules:
- Output ONLY the SQL statement, no explanation, no markdown fences.
- Use standard DuckDB syntax and quote identifiers with double quotes.
- Always use a SELECT statement; never modify the data.
- If the question cannot be answered from the available columns, output the literal: NO_INFO_AVAILABLE""",
        ),
        ("human", "{question}"),
    ]
)

SQL_FIX_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert data analyst fixing a broken DuckDB SQL query.

Database schema:
{schema}

The previous query:
{previous_sql}

It failed with this error:
{sql_error}

Output ONLY a corrected SQL statement, no explanation, no markdown fences.
If the query genuinely cannot be fixed, output the literal: NO_INFO_AVAILABLE""",
        ),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a concise data analyst. Answer the user's question using ONLY the provided query results.

Question: {question}

SQL used:
{sql}

Query result columns: {columns}

Query result rows:
{rows}

Answer in 2-4 sentences with specific numbers. If the results are empty or do not
answer the question, reply with exactly: {no_info_message}""",
        ),
        ("human", "Provide the answer."),
    ]
)

_CHART_KEYWORDS = {
    "histogram": ["distribution", "histogram", "frequency", "spread", "range"],
    "scatter": ["correlation", "scatter", "relationship", "vs", "versus", "compare"],
    "line": ["trend", "time", "line", "over time", "period", "growth", "change"],
    "bar": ["top", "highest", "largest", "most", "by", "category", "group", "total", "sum"],
    "pie": ["pie", "proportion", "percentage", "share"],
    "box": ["box", "outlier", "quartile", "median"],
}


class AnalystStateUpdate:
    """Helper to keep node return values consistent."""

    @staticmethod
    def no_info(reason: str) -> Dict[str, Any]:
        return {
            "answer": NO_INFO_MESSAGE,
            "no_info": reason,
            "sql_query": None,
        }

    @staticmethod
    def answer(answer: str) -> Dict[str, Any]:
        return {"answer": answer, "no_info": None}


def _get_llm():
    """Create the Gemini chat model, or None when no API key is configured."""
    if not HAS_API_KEY or ChatGoogleGenerativeAI is None:
        return None
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
    )


def _extract_sql(text: str) -> Optional[str]:
    """Extract the SQL statement from an LLM response."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = cleaned.rstrip(";").strip()
    lowered = cleaned.lower()
    if "no_info_available" in lowered:
        return None
    if lowered.startswith("select") or lowered.startswith("with"):
        return cleaned
    return None


def generate_sql(state: AnalystState) -> Dict[str, Any]:
    """Node: turn the natural language question into a SQL query."""
    llm = _get_llm()
    if llm is None:
        return AnalystStateUpdate.no_info("Gemini API key not configured")

    chain = SQL_GENERATION_PROMPT | llm | StrOutputParser()
    response = chain.invoke(
        {"schema": state["schema"], "question": state["question"]}
    )

    sql = _extract_sql(response)
    if sql is None:
        return AnalystStateUpdate.no_info("No SQL could be generated for the question")

    return {"sql_query": sql, "no_info": None, "sql_error": None, "retry_count": 0}


def execute_sql(state: AnalystState) -> Dict[str, Any]:
    """Node: execute the SQL query against the data via DuckDB."""
    engine = state.get("engine")
    if engine is None:
        return {"sql_error": "SQL engine not initialized", "no_info": "SQL engine not available"}

    query = state.get("sql_query")
    if not query:
        return {"no_info": "No SQL query to execute"}

    try:
        columns, rows = engine.execute(query)
    except Exception as exc:  # noqa: BLE001
        return {
            "sql_error": str(exc),
            "no_info": None,
            "columns": None,
            "rows": None,
        }

    if not rows:
        return {
            "columns": columns,
            "rows": [],
            "sql_error": None,
            "no_info": "The query returned no rows",
        }

    return {"columns": columns, "rows": rows, "sql_error": None, "no_info": None}


def fix_sql(state: AnalystState) -> Dict[str, Any]:
    """Node: retry SQL generation with the error message as context."""
    llm = _get_llm()
    if llm is None:
        return AnalystStateUpdate.no_info("Gemini API key not configured")

    retry_count = int(state.get("retry_count", 0))
    if retry_count >= MAX_SQL_RETRIES:
        return {
            "no_info": f"SQL generation failed after {MAX_SQL_RETRIES} attempts",
            "answer": NO_INFO_MESSAGE,
            "sql_query": None,
        }

    chain = SQL_FIX_PROMPT | llm | StrOutputParser()
    response = chain.invoke(
        {
            "schema": state["schema"],
            "question": state["question"],
            "previous_sql": state.get("sql_query", ""),
            "sql_error": state.get("sql_error", ""),
        }
    )

    sql = _extract_sql(response)
    if sql is None:
        return {
            "no_info": "SQL could not be corrected",
            "answer": NO_INFO_MESSAGE,
            "sql_query": None,
        }

    return {
        "sql_query": sql,
        "retry_count": retry_count + 1,
        "sql_error": None,
        "no_info": None,
    }


def generate_answer(state: AnalystState) -> Dict[str, Any]:
    """Node: generate a natural language answer from the query results."""
    rows = state.get("rows")
    if not rows:
        return AnalystStateUpdate.no_info("The query returned no rows")

    llm = _get_llm()
    if llm is None:
        # Deterministic fallback so the app still works without an API key.
        columns = state.get("columns") or []
        summary = ", ".join(columns[:3]) or "rows"
        answer = f"The query returned {len(rows)} result(s) with columns: {summary}."
        return AnalystStateUpdate.answer(answer)

    chain = ANSWER_PROMPT | llm | StrOutputParser()
    response = chain.invoke(
        {
            "question": state["question"],
            "sql": state.get("sql_query", ""),
            "columns": ", ".join(state.get("columns") or []),
            "rows": rows,
            "no_info_message": NO_INFO_MESSAGE,
        }
    )

    if NO_INFO_MESSAGE.lower() in response.strip().lower():
        return AnalystStateUpdate.no_info("The answer node could not answer from the data")
    return AnalystStateUpdate.answer(response.strip())


def suggest_chart(state: AnalystState) -> Dict[str, Any]:
    """Node: heuristically suggest a chart type for the question."""
    question = state.get("question", "").lower()
    chart_type: Optional[str] = None
    for candidate, keywords in _CHART_KEYWORDS.items():
        if any(kw in question for kw in keywords):
            chart_type = candidate
            break

    columns = state.get("columns") or []
    numeric = state.get("numeric_columns") or []
    categorical = state.get("categorical_columns") or []

    chart_columns: List[str] = []
    if chart_type == "histogram" and numeric:
        chart_columns = [numeric[0]]
    elif chart_type == "scatter" and len(numeric) >= 2:
        chart_columns = numeric[:2]
    elif chart_type == "line":
        chart_columns = (categorical[:1] + numeric[:1]) or columns[:2]
    elif chart_type in ("bar", "pie"):
        chart_columns = (categorical[:1] + numeric[:1]) or columns[:2]
    elif chart_type == "box" and numeric:
        chart_columns = numeric[:3]

    return {"chart_suggestion": chart_type, "chart_columns": chart_columns}


def route_after_execute(state: AnalystState) -> str:
    """Route to fix_sql, no_info, or generate_answer after execution."""
    if state.get("sql_error"):
        retry_count = int(state.get("retry_count", 0))
        if retry_count < MAX_SQL_RETRIES:
            return "fix_sql"
        return "no_info"
    if state.get("no_info") or not state.get("rows"):
        return "no_info"
    return "generate_answer"


def route_after_fix(state: AnalystState) -> str:
    """Route back to execution, or to no_info when the fix failed."""
    if state.get("sql_query") and not state.get("no_info"):
        return "execute_sql"
    return "no_info"


def no_info(state: AnalystState) -> Dict[str, Any]:
    """Node: final fallback that reports no information is available."""
    reason = state.get("no_info") or "No information could be derived from the data"
    return {
        "answer": NO_INFO_MESSAGE,
        "no_info": reason,
        "sql_query": None,
        "rows": None,
    }
