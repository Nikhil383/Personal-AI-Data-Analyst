from __future__ import annotations

from pathlib import Path
import pandas as pd

from .tools.data_tools import answer_question
from .tools.web_tools import web_research


def run_analysis(df: pd.DataFrame, question: str, output_dir: Path, include_research: bool = False) -> dict:
    """Crew-compatible orchestration boundary.

    Local analysis always runs; install/configure CrewAI and Gemini to replace this
    deterministic implementation with an LLM-led crew without changing the API.
    """
    result = answer_question(df, question, output_dir)
    research = web_research(question) if include_research else None
    summary = f"Analysis of {result['profile']['rows']:,} rows and {result['profile']['columns']} columns. " + " ".join(result["insights"][:2])
    recommendations = [
        "Validate missing values before using the affected fields in operational decisions.",
        "Use the displayed trend and summary metrics to investigate the largest changes by segment.",
    ]
    return {**result, "summary": summary, "recommendations": recommendations, "research": research}
