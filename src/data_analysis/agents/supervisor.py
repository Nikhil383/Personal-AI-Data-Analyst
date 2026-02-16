"""
Supervisor Agent - Routes queries to appropriate specialized agents.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from pydantic import BaseModel, Field

from ..config import Config


class RouteDecision(BaseModel):
    """Structured output for routing decisions."""
    agent: Literal["data_cleaner", "stats_analyst", "visualizer", "query_answerer", "end"] = Field(
        description="The agent to route to, or 'end' if the query is complete"
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen"
    )


class SupervisorAgent:
    """
    Supervisor agent that analyzes queries and routes to specialized agents.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the supervisor agent."""
        api_key = api_key or Config.GOOGLE_API_KEY
        
        self.llm = ChatGoogleGenerativeAI(
            model=Config.SUPERVISOR_MODEL,
            google_api_key=api_key,
            temperature=Config.SUPERVISOR_TEMPERATURE,
        )
        
        # Create structured output LLM
        self.structured_llm = self.llm.with_structured_output(RouteDecision)
        
        self.system_prompt = """You are a supervisor agent coordinating a team of data analysis specialists.

Your team consists of:
1. **data_cleaner**: Handles data cleaning tasks (missing values, duplicates, outliers, data type fixes)
2. **stats_analyst**: Performs statistical analysis (correlations, hypothesis testing, distributions, trends)
3. **visualizer**: Creates charts and visualizations (plots, graphs, heatmaps)
4. **query_answerer**: Answers general questions about the data (summaries, insights, explanations)

Your job is to:
1. Analyze the user's query
2. Determine which specialist agent should handle it
3. Route to that agent
4. If the task is complete or no agent is needed, route to 'end'

Guidelines:
- If the query mentions "clean", "missing", "duplicates", "outliers" → data_cleaner
- If the query asks for "correlation", "statistics", "test", "distribution" → stats_analyst
- If the query asks to "plot", "chart", "visualize", "graph", "show" → visualizer
- For general questions, summaries, or insights → query_answerer
- If responding to a completed task or greeting → end

Be decisive and choose the most appropriate agent."""
    
    def route(self, query: str, conversation_history: list = None) -> RouteDecision:
        """
        Route a query to the appropriate agent.
        
        Args:
            query: The user's query
            conversation_history: Optional conversation history for context
            
        Returns:
            RouteDecision with agent name and reasoning
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"User query: {query}\n\nWhich agent should handle this?")
        ]
        
        decision = self.structured_llm.invoke(messages)
        return decision
