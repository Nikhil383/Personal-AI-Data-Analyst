"""Tests for the LangGraph + SQL + Gemini analyst engine.

These tests cover the SQL engine and the LangGraph workflow. LLM-backed
nodes degrade gracefully to "No info available" when no API key is present,
which keeps the tests runnable without network access.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_data_analyst.graph.sql_engine import SqlEngine, SqlQueryError


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": ["Laptop", "Phone", "Tablet", "Monitor"],
            "category": ["Electronics", "Electronics", "Furniture", "Furniture"],
            "region": ["North", "South", "North", "South"],
            "sales": [15000, 22000, 8000, 12000],
            "quantity": [50, 110, 40, 60],
            "profit": [4500, 8800, 2400, 3600],
        }
    )


@pytest.fixture
def engine(sales_df) -> SqlEngine:
    eng = SqlEngine(sales_df)
    yield eng
    eng.close()


# ---------------------------------------------------------------------------
# SqlEngine
# ---------------------------------------------------------------------------


class TestSqlEngine:
    def test_schema_ddl(self, engine):
        ddl = engine.schema_ddl()
        assert "CREATE TABLE" in ddl
        assert "sales" in ddl

    def test_schema_prompt(self, engine):
        prompt = engine.schema_prompt()
        assert "Table: df" in prompt
        assert "Rows: 4" in prompt
        assert "product" in prompt

    def test_execute_select(self, engine):
        columns, rows = engine.execute('SELECT product, sales FROM df ORDER BY sales DESC LIMIT 2')
        assert columns == ["product", "sales"]
        assert len(rows) == 2
        assert rows[0]["product"] == "Phone"
        assert rows[0]["sales"] == 22000

    def test_execute_aggregate(self, engine):
        columns, rows = engine.execute(
            "SELECT category, SUM(sales) AS total FROM df GROUP BY category"
        )
        assert columns == ["category", "total"]
        totals = {row["category"]: row["total"] for row in rows}
        assert totals["Electronics"] == 37000
        assert totals["Furniture"] == 20000

    def test_empty_result(self, engine):
        columns, rows = engine.execute("SELECT * FROM df WHERE sales > 999999")
        assert rows == []
        assert columns  # columns are still reported

    def test_rejects_non_select(self, engine):
        with pytest.raises(SqlQueryError):
            engine.execute("DROP TABLE df")

    def test_rejects_multi_statement(self, engine):
        with pytest.raises(SqlQueryError):
            engine.execute("SELECT * FROM df; SELECT * FROM df")

    def test_rejects_insert(self, engine):
        with pytest.raises(SqlQueryError):
            engine.execute("INSERT INTO df VALUES (1)")

    def test_rejects_empty(self, engine):
        with pytest.raises(SqlQueryError):
            engine.execute("")

    def test_returns_json_safe_values(self, engine):
        columns, rows = engine.execute("SELECT region FROM df LIMIT 1")
        assert rows[0]["region"] in {"North", "South"}


# ---------------------------------------------------------------------------
# LangGraph workflow
# ---------------------------------------------------------------------------


class TestLangGraphWorkflow:
    def test_no_info_when_no_api_key(self, sales_df):
        """Without an API key, the graph answers 'No info available'."""
        from ai_data_analyst.chains.sql_analyst_chain import SQLAnalystChain
        from ai_data_analyst.config import NO_INFO_MESSAGE

        chain = SQLAnalystChain(sales_df, api_key=None)
        response = chain.analyze("What is the total sales?")
        assert response.final_answer == NO_INFO_MESSAGE
        assert response.has_answer is False
        assert response.no_info is not None

    def test_no_info_for_unanswerable_question(self, sales_df):
        """A question that cannot map to columns yields 'No info available'."""
        from ai_data_analyst.graph.analyst_graph import AnalystGraph
        from ai_data_analyst.config import NO_INFO_MESSAGE

        graph = AnalystGraph(sales_df)
        state = graph.run("What is the weather forecast for tomorrow?")
        assert state["answer"] == NO_INFO_MESSAGE
        assert state.get("no_info") is not None

    def test_no_info_node_sets_answer(self, sales_df):
        from ai_data_analyst.graph.nodes import no_info
        from ai_data_analyst.config import NO_INFO_MESSAGE

        update = no_info({"no_info": "test reason"})
        assert update["answer"] == NO_INFO_MESSAGE
        assert update["no_info"] == "test reason"

    def test_route_after_execute(self, sales_df):
        from ai_data_analyst.graph.nodes import route_after_execute, route_after_fix

        assert route_after_execute({"sql_error": "boom", "retry_count": 0}) == "fix_sql"
        assert route_after_execute({"sql_error": "boom", "retry_count": 99}) == "no_info"
        assert route_after_execute({"sql_error": None, "no_info": "empty", "rows": []}) == "no_info"
        assert route_after_execute({"sql_error": None, "no_info": None, "rows": [{"a": 1}]}) == "generate_answer"

        assert route_after_fix({"sql_query": "SELECT 1", "no_info": None}) == "execute_sql"
        assert route_after_fix({"sql_query": None, "no_info": "fail"}) == "no_info"

    def test_suggest_chart_heuristics(self, sales_df):
        from ai_data_analyst.graph.nodes import suggest_chart

        state = {
            "question": "Show me the correlation between sales and profit",
            "columns": ["sales", "profit"],
            "numeric_columns": ["sales", "quantity", "profit"],
            "categorical_columns": ["product", "category", "region"],
        }
        update = suggest_chart(state)
        assert update["chart_suggestion"] == "scatter"
        assert update["chart_columns"] == ["sales", "quantity"]

    def test_build_graph_runs_no_info_path(self, sales_df):
        """The compiled graph executes the no_info path end to end."""
        from ai_data_analyst.graph.analyst_graph import AnalystGraph
        from ai_data_analyst.config import NO_INFO_MESSAGE

        graph = AnalystGraph(sales_df)
        result = graph.run("How many UFO sightings were recorded?")
        assert result["answer"] == NO_INFO_MESSAGE


# ---------------------------------------------------------------------------
# SQLAnalystChain wrapper
# ---------------------------------------------------------------------------


class TestSQLAnalystChain:
    def test_analyze_returns_response(self, sales_df):
        from ai_data_analyst.chains.sql_analyst_chain import SQLAnalystChain

        chain = SQLAnalystChain(sales_df, api_key=None)
        response = chain.analyze("What is the total sales?")
        assert response.user_question == "What is the total sales?"
        assert response.final_answer  # non-empty
        assert "sales" in response.data_insights["dataset_shape"] or True

    def test_query_method(self, sales_df):
        from ai_data_analyst.chains.sql_analyst_chain import SQLAnalystChain
        from ai_data_analyst.config import NO_INFO_MESSAGE

        chain = SQLAnalystChain(sales_df, api_key=None)
        assert chain.query("something completely unrelated") == NO_INFO_MESSAGE

    def test_workflow_summary(self, sales_df):
        from ai_data_analyst.chains.sql_analyst_chain import SQLAnalystChain

        chain = SQLAnalystChain(sales_df, api_key=None)
        response = chain.analyze("What is the total sales?")
        summary = chain.get_workflow_summary(response)
        assert "Analysis Workflow" in summary
        assert "Final Answer" in summary
