from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px


def _json(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return str(value)


SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".ndjson", ".parquet"}


def load_tabular(path: Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load a supported flat tabular file into a dataframe."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TABULAR_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_TABULAR_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {allowed}")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".ndjson"}:
        return pd.read_json(path, lines=suffix == ".ndjson")
    return _load_delimited(path, "\t" if suffix == ".tsv" else None)


def _load_delimited(path: Path, separator: str | None = None) -> pd.DataFrame:
    """Read common delimited-text encodings and return a normalized dataframe."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            options: dict[str, Any] = {"encoding": encoding}
            if separator is None:
                options.update({"sep": None, "engine": "python"})
            else:
                options["sep"] = separator
            return pd.read_csv(path, **options)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep=separator or ",")


def load_csv(path: Path) -> pd.DataFrame:
    """Backward-compatible CSV loader."""
    return load_tabular(path)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum()
    duplicates = int(df.duplicated().sum())
    columns = []
    for name in df.columns:
        series = df[name]
        columns.append({
            "name": str(name),
            "dtype": str(series.dtype),
            "missing": int(missing[name]),
            "missing_pct": round(float(missing[name] / max(len(df), 1) * 100), 1),
            "unique": int(series.nunique(dropna=True)),
            "sample": [_json(v) for v in series.dropna().head(3).tolist()],
        })
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)), "duplicates": duplicates,
        "missing_cells": int(missing.sum()), "column_profile": columns,
        "preview": json.loads(df.head(8).to_json(orient="records", date_format="iso", default_handler=_json)),
    }


def _date_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if re.search(r"date|time|month|year|quarter", str(c), re.I)]
    for column in candidates:
        converted = pd.to_datetime(df[column], errors="coerce")
        if converted.notna().mean() > 0.6:
            return str(column)
    return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.select_dtypes(include="number").columns]


def answer_question(df: pd.DataFrame, question: str, output_dir: Path) -> dict[str, Any]:
    """Deterministic local analysis used as the data analyst tool and safe fallback."""
    numeric = _numeric_columns(df)
    question_lower = question.lower()
    date_col = _date_column(df)
    metric_col = next((c for c in numeric if c.lower() in question_lower), numeric[0] if numeric else None)
    insights: list[str] = []
    metrics: list[dict[str, Any]] = []
    chart_path: str | None = None

    if metric_col:
        series = df[metric_col].dropna()
        metrics.extend([
            {"label": f"Total {metric_col}", "value": f"{series.sum():,.2f}"},
            {"label": f"Average {metric_col}", "value": f"{series.mean():,.2f}"},
            {"label": f"Range", "value": f"{series.min():,.2f} – {series.max():,.2f}"},
        ])
        insights.append(f"{metric_col} totals {series.sum():,.2f} across {len(series):,} populated records, with an average of {series.mean():,.2f}.")
        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            trend = pd.DataFrame({"date": dates, "value": df[metric_col]}).dropna().sort_values("date")
            if len(trend) > 1:
                grouped = trend.groupby(pd.Grouper(key="date", freq="ME"))["value"].sum().reset_index()
                if len(grouped) > 1:
                    change = (grouped.value.iloc[-1] / grouped.value.iloc[0] - 1) * 100 if grouped.value.iloc[0] else 0
                    direction = "increased" if change >= 0 else "decreased"
                    insights.append(f"Monthly {metric_col} {direction} {abs(change):.1f}% from the first to last observed month.")
                    figure = px.line(grouped, x="date", y="value", markers=True, title=f"{metric_col} over time")
                    chart_file = output_dir / "trend.html"
                    figure.write_html(chart_file, include_plotlyjs="cdn")
                    chart_path = "/outputs/trend.html"
        elif len(numeric) > 1:
            insights.append(f"Other numeric fields available for relationship analysis: {', '.join(c for c in numeric if c != metric_col)[:120]}.")
    else:
        insights.append("The file has no numeric columns, so the analysis focuses on data quality and categorical composition.")

    profile = profile_dataframe(df)
    if profile["missing_cells"]:
        insights.append(f"Data quality note: {profile['missing_cells']:,} cells are missing; treat related comparisons cautiously.")
    if profile["duplicates"]:
        insights.append(f"Data quality note: {profile['duplicates']:,} duplicate rows were detected.")
    return {"metrics": metrics, "insights": insights, "chart_url": chart_path, "profile": profile}
