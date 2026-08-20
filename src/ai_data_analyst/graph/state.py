"""LangGraph state definition for the SQL-based data analyst workflow."""
from typing import Any, Dict, List, Optional, TypedDict

from ai_data_analyst.graph.sql_engine import SqlEngine


class AnalystState(TypedDict, total=False):
    """State passed between nodes of the analyst LangGraph."""

    question: str
    schema: str
    engine: SqlEngine
    numeric_columns: List[str]
    categorical_columns: List[str]
    sql_query: Optional[str]
    sql_error: Optional[str]
    rows: Optional[List[Dict[str, Any]]]
    columns: Optional[List[str]]
    answer: Optional[str]
    chart_suggestion: Optional[str]
    chart_columns: Optional[List[str]]
    no_info: Optional[str]
    retry_count: int
    messages: List[Dict[str, Any]]
