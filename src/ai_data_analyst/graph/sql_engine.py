"""SQL execution engine backed by DuckDB.

Loads a pandas DataFrame into an in-memory DuckDB instance and exposes a
safe SELECT-only execution API for the LangGraph text-to-SQL workflow.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

# Statements that are never allowed, regardless of position in the query.
_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "copy",
    "grant",
    "revoke",
    "vacuum",
    "call",
    "install",
    "load",
)


class SqlQueryError(Exception):
    """Raised when a SQL query is rejected by validation or execution."""


class SqlEngine:
    """Executes read-only SELECT queries against a DataFrame via DuckDB."""

    def __init__(self, df: pd.DataFrame, table_name: str = "df"):
        self.df = df
        self.table_name = table_name
        self._conn = duckdb.connect(database=":memory:")
        # Register the dataframe as a view so queries reference `df`.
        self._conn.register(table_name, df)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Access the underlying DuckDB connection."""
        return self._conn

    def schema_ddl(self) -> str:
        """Return a compact schema description used to prompt the LLM."""
        parts = [f"CREATE TABLE {self.table_name} ("]
        lines = []
        for col, dtype in self.df.dtypes.items():
            lines.append(f"  {_quote_ident(col)} {_to_sql_type(dtype)}")
        parts.append(",\n".join(lines))
        parts.append(");")
        return "\n".join(parts)

    def schema_prompt(self, max_sample_rows: int = 5) -> str:
        """Human/machine readable schema including column types and samples."""
        lines = [f"Table: {self.table_name}", f"Rows: {len(self.df)}"]
        lines.append("Columns:")
        for col in self.df.columns:
            sample = _format_sample(self.df[col], max_sample_rows)
            lines.append(f"  - {col} ({self.df[col].dtype}): {sample}")
        return "\n".join(lines)

    def validate(self, query: str) -> None:
        """Validate a query is a single read-only SELECT statement."""
        if not query or not query.strip():
            raise SqlQueryError("Empty SQL query")

        # Strip trailing semicolon and ensure exactly one statement.
        stripped = query.strip().rstrip(";").strip()
        if ";" in stripped:
            raise SqlQueryError("Multiple SQL statements are not allowed")

        lowered = stripped.lower()

        # Must begin with SELECT (or WITH, for CTEs).
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise SqlQueryError("Only SELECT queries are allowed")

        # Reject any forbidden keyword anywhere in the statement.
        for keyword in _FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", lowered):
                raise SqlQueryError(f"Forbidden SQL keyword: {keyword}")

    def execute(self, query: str, max_rows: int = 200) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Validate and execute a SELECT query, returning (columns, rows)."""
        self.validate(query)
        try:
            result = self._conn.execute(query).fetchall()
        except Exception as exc:  # noqa: BLE001 - surface DuckDB errors to the LLM
            raise SqlQueryError(f"SQL execution failed: {exc}") from exc

        cursor = self._conn.execute(query)
        columns = [desc[0] for desc in cursor.description]

        rows: List[Dict[str, Any]] = []
        for record in result[:max_rows]:
            rows.append({col: _json_safe(value) for col, value in zip(columns, record)})
        return columns, rows

    def get_table_names(self) -> List[str]:
        """Return the list of tables available in this engine."""
        tables = self._conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in tables]

    def close(self) -> None:
        """Close the underlying connection."""
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


def _quote_ident(ident: str) -> str:
    """Quote an identifier with double quotes (DuckDB style)."""
    return f'"{ident.replace(chr(34), chr(34) + chr(34))}"'


def _to_sql_type(dtype) -> str:
    """Map a pandas dtype to a DuckDB type name."""
    dtype = str(dtype).lower()
    if "int" in dtype:
        return "BIGINT"
    if "float" in dtype:
        return "DOUBLE"
    if "bool" in dtype:
        return "BOOLEAN"
    if "datetime" in dtype or "date" in dtype:
        return "TIMESTAMP"
    return "VARCHAR"


def _format_sample(series: pd.Series, limit: int) -> str:
    """Format a small sample of column values for the schema prompt."""
    sample = series.dropna().head(limit).tolist()
    if not sample:
        return "no values"
    return ", ".join(repr(v) for v in sample)


def _json_safe(value: Any) -> Any:
    """Convert a value into something JSON-serializable."""
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalars
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "isoformat"):  # datetime / date
        return value.isoformat()
    if isinstance(value, (dict, list)):
        try:
            return json.loads(json.dumps(value, default=str))
        except Exception:  # noqa: BLE001
            return str(value)
    return value
