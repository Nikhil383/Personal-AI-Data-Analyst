"""Definitions for the three roles in the planned CrewAI deployment.

They are kept as role specifications so the application remains useful without
cloud credentials; a production CrewAI adapter can consume these unchanged.
"""

DATA_ANALYST = {
    "role": "Data Analyst",
    "goal": "Profile CSV data, calculate metrics, identify trends, and flag quality issues.",
    "tools": ["analyze_csv", "query_csv", "save_chart"],
}

RESEARCH_ANALYST = {
    "role": "Research Analyst",
    "goal": "Add clearly separated, current external context only when it is needed.",
    "tools": ["web_research"],
}

REPORT_ANALYST = {
    "role": "Report Analyst",
    "goal": "Translate analysis into an executive summary, recommendations, and caveats.",
    "tools": [],
}
