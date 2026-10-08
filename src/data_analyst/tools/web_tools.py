def web_research(query: str) -> str:
    """Extension point for Nimble Web Search Agent.

    The local app intentionally does not invent external facts when no configured
    research provider is available.
    """
    return f"External research was not run for: {query}. Configure NIMBLE_API_KEY to enable a research provider."
