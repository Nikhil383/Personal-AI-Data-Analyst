"""
State schema for the multi-agent data analysis workflow.
"""
from typing import TypedDict, Annotated, Sequence, Optional, Any
from langchain_core.messages import BaseMessage
import operator
import pandas as pd


class AgentState(TypedDict):
    """
    Shared state for the multi-agent workflow.
    
    This state is passed between all agents in the LangGraph workflow.
    """
    # Conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Current dataframe being analyzed
    dataframe: Optional[pd.DataFrame]
    
    # Metadata about the dataframe
    dataframe_info: dict[str, Any]
    
    # Currently active agent
    current_agent: str
    
    # Intermediate analysis results from agents
    analysis_results: dict[str, Any]
    
    # Generated visualizations (as base64 or file paths)
    visualizations: list[str]
    
    # Next action to take (for routing)
    next_action: str
    
    # User's original query
    user_query: str
    
    # Final response to user
    final_response: Optional[str]
