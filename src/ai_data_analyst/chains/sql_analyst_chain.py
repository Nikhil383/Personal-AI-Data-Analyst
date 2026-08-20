"""SQLAnalystChain - LangGraph + SQL + Gemini powered analyst chain.

Replaces the older pandas ReAct agent with a modern text-to-SQL workflow:
    Natural Language -> LangGraph -> Gemini (generate SQL) -> DuckDB (execute)
    -> Gemini (generate natural language answer)

When no information can be derived (no API key, invalid SQL, empty results,
or an unanswerable question) the chain answers "No info available".
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ai_data_analyst.config import NO_INFO_MESSAGE
from ai_data_analyst.graph.analyst_graph import AnalystGraph


@dataclass
class SQLAnalysisResponse:
    """Structured response produced by the LangGraph SQL analyst."""

    user_question: str
    final_answer: str
    sql_query: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    chart_type: Optional[str] = None
    chart_columns: Optional[List[str]] = None
    no_info: Optional[str] = None
    data_insights: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_answer(self) -> bool:
        """True when the response contains a real answer, not just a fallback."""
        return bool(self.final_answer) and self.final_answer.strip() != NO_INFO_MESSAGE


class SQLAnalystChain:
    """LangGraph-based data analyst chain using text-to-SQL and Gemini."""

    def __init__(self, df: pd.DataFrame, api_key: Optional[str] = None, model: Optional[str] = None):
        self.df = df
        self.api_key = api_key
        self.model = model
        self.graph = AnalystGraph(df)

    def analyze(self, user_query: str) -> SQLAnalysisResponse:
        """Run the full LangGraph workflow for a question."""
        state = self.graph.run(user_query)

        return SQLAnalysisResponse(
            user_question=user_query,
            final_answer=state.get("answer") or NO_INFO_MESSAGE,
            sql_query=state.get("sql_query"),
            rows=state.get("rows"),
            columns=state.get("columns"),
            chart_type=state.get("chart_suggestion"),
            chart_columns=state.get("chart_columns"),
            no_info=state.get("no_info"),
            data_insights=self._extract_insights(),
        )

    def query(self, user_query: str) -> str:
        """Convenience method returning just the answer string."""
        response = self.analyze(user_query)
        return response.final_answer

    def get_workflow_summary(self, response: SQLAnalysisResponse) -> str:
        """Generate a human-readable summary of the LangGraph + SQL workflow."""
        lines = [
            "📊 Analysis Workflow",
            "=" * 50,
            "",
            f"📝 User Question: {response.user_question}",
            f"🗄️ SQL Query: {response.sql_query or 'No SQL generated'}",
            f"📋 Result Rows: {len(response.rows) if response.rows is not None else 0}",
            "",
            "✅ Final Answer:",
            response.final_answer,
        ]
        if response.chart_type:
            lines.append("")
            lines.append(f"📈 Suggested Chart: {response.chart_type}")
            if response.chart_columns:
                lines.append(f"   Columns: {', '.join(response.chart_columns)}")
        if response.no_info:
            lines.append("")
            lines.append(f"ℹ️ Note: {response.no_info}")
        return "\n".join(lines)

    def _extract_insights(self) -> Dict[str, Any]:
        """Extract basic dataset insights."""
        numeric = self.df.select_dtypes(include=["number"]).columns
        return {
            "dataset_shape": f"{len(self.df)} rows x {len(self.df.columns)} columns",
            "numeric_columns": len(numeric),
            "categorical_columns": len(
                self.df.select_dtypes(include=["object", "string", "category"]).columns
            ),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
        }
