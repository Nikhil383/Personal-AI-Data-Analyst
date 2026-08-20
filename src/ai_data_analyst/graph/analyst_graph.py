"""LangGraph workflow builder for the text-to-SQL data analyst.

Graph structure::

    START -> generate_sql -> execute_sql
                                   |--(sql_error & retries left)-> fix_sql -> execute_sql
                                   |--(no rows / exhausted)-> no_info
                                   `--(ok)-> generate_answer -> suggest_chart -> END

Any node that cannot produce an answer routes to ``no_info`` which answers
"No info available".
"""
from typing import Any, Dict, Optional

import pandas as pd
from langgraph.graph import END, START, StateGraph

from ai_data_analyst.graph.nodes import (
    execute_sql,
    fix_sql,
    generate_answer,
    generate_sql,
    no_info,
    route_after_execute,
    route_after_fix,
    suggest_chart,
)
from ai_data_analyst.graph.sql_engine import SqlEngine
from ai_data_analyst.graph.state import AnalystState


class AnalystGraph:
    """Compiles and runs the LangGraph analyst workflow."""

    def __init__(self, df: pd.DataFrame, schema_override: Optional[str] = None):
        self.df = df
        self.engine = SqlEngine(df)
        self.schema_override = schema_override

    @property
    def schema(self) -> str:
        if self.schema_override:
            return self.schema_override
        return self.engine.schema_prompt()

    def build(self):
        """Build and compile the state graph."""
        graph = StateGraph(AnalystState)

        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_sql", execute_sql)
        graph.add_node("fix_sql", fix_sql)
        graph.add_node("generate_answer", generate_answer)
        graph.add_node("suggest_chart", suggest_chart)
        graph.add_node("no_info", no_info)

        graph.add_edge(START, "generate_sql")
        graph.add_edge("generate_sql", "execute_sql")

        graph.add_conditional_edges(
            "execute_sql",
            route_after_execute,
            {"fix_sql": "fix_sql", "no_info": "no_info", "generate_answer": "generate_answer"},
        )
        graph.add_conditional_edges(
            "fix_sql",
            route_after_fix,
            {"execute_sql": "execute_sql", "no_info": "no_info"},
        )

        graph.add_edge("generate_answer", "suggest_chart")
        graph.add_edge("suggest_chart", END)
        graph.add_edge("no_info", END)

        return graph.compile()

    def run(self, question: str) -> Dict[str, Any]:
        """Run the graph for a question and return the final state."""
        compiled = self.build()
        initial_state: Dict[str, Any] = {
            "question": question,
            "schema": self.schema,
            "engine": self.engine,
            "numeric_columns": list(self.df.select_dtypes(include=["number"]).columns),
            "categorical_columns": list(
                self.df.select_dtypes(include=["object", "string", "category"]).columns
            ),
            "columns": None,
            "rows": None,
            "sql_query": None,
            "sql_error": None,
            "answer": None,
            "chart_suggestion": None,
            "chart_columns": None,
            "no_info": None,
            "retry_count": 0,
            "messages": [],
        }
        return compiled.invoke(initial_state)
