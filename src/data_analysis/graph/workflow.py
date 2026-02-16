"""
LangGraph Workflow - Multi-agent orchestration for data analysis.
"""
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
from typing import Optional

from .state import AgentState
from ..agents.supervisor import SupervisorAgent
from ..agents.data_cleaner import DataCleanerAgent
from ..agents.stats_analyst import StatsAnalystAgent
from ..agents.visualizer import VisualizerAgent
from ..agents.query_answerer import QueryAnswererAgent
from ..config import Config


class AgenticWorkflow:
    """
    Multi-agent workflow using LangGraph.
    """
    
    def __init__(self, df: pd.DataFrame, api_key: str = None):
        """
        Initialize the agentic workflow.
        
        Args:
            df: The pandas DataFrame to analyze
            api_key: Google API key (optional, will use Config if not provided)
        """
        self.df = df
        self.api_key = api_key or Config.GOOGLE_API_KEY
        
        # Initialize all agents
        self.supervisor = SupervisorAgent(self.api_key)
        self.data_cleaner = DataCleanerAgent(self.api_key)
        self.stats_analyst = StatsAnalystAgent(self.api_key)
        self.visualizer = VisualizerAgent(self.api_key)
        self.query_answerer = QueryAnswererAgent(self.api_key)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("data_cleaner", self._data_cleaner_node)
        workflow.add_node("stats_analyst", self._stats_analyst_node)
        workflow.add_node("visualizer", self._visualizer_node)
        workflow.add_node("query_answerer", self._query_answerer_node)
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor to agents
        workflow.add_conditional_edges(
            "supervisor",
            self._route_query,
            {
                "data_cleaner": "data_cleaner",
                "stats_analyst": "stats_analyst",
                "visualizer": "visualizer",
                "query_answerer": "query_answerer",
                "end": END
            }
        )
        
        # All agents return to supervisor (or could go to END)
        workflow.add_edge("data_cleaner", END)
        workflow.add_edge("stats_analyst", END)
        workflow.add_edge("visualizer", END)
        workflow.add_edge("query_answerer", END)
        
        return workflow.compile()
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor node - routes to appropriate agent."""
        query = state["user_query"]
        
        # Get routing decision
        decision = self.supervisor.route(query)
        
        # Update state
        state["current_agent"] = "supervisor"
        state["next_action"] = decision.agent
        
        # Add supervisor's reasoning to messages
        state["messages"].append(
            AIMessage(content=f"[Supervisor] Routing to {decision.agent}: {decision.reasoning}")
        )
        
        return state
    
    def _data_cleaner_node(self, state: AgentState) -> AgentState:
        """Data cleaner node."""
        query = state["user_query"]
        df = state["dataframe"] or self.df
        
        result = self.data_cleaner.clean(df, query)
        
        # Update state
        state["current_agent"] = "data_cleaner"
        state["dataframe"] = result.get("dataframe", df)
        state["final_response"] = result["output"]
        state["analysis_results"]["data_cleaner"] = result
        
        # Add to messages
        state["messages"].append(
            AIMessage(content=f"[Data Cleaner] {result['output']}")
        )
        
        return state
    
    def _stats_analyst_node(self, state: AgentState) -> AgentState:
        """Stats analyst node."""
        query = state["user_query"]
        df = state["dataframe"] or self.df
        
        result = self.stats_analyst.analyze(df, query)
        
        # Update state
        state["current_agent"] = "stats_analyst"
        state["final_response"] = result["output"]
        state["analysis_results"]["stats_analyst"] = result
        
        # Add to messages
        state["messages"].append(
            AIMessage(content=f"[Stats Analyst] {result['output']}")
        )
        
        return state
    
    def _visualizer_node(self, state: AgentState) -> AgentState:
        """Visualizer node."""
        query = state["user_query"]
        df = state["dataframe"] or self.df
        
        result = self.visualizer.visualize(df, query)
        
        # Update state
        state["current_agent"] = "visualizer"
        state["final_response"] = result["output"]
        state["analysis_results"]["visualizer"] = result
        
        if result.get("has_plot"):
            state["visualizations"].append("current_plot")
        
        # Add to messages
        state["messages"].append(
            AIMessage(content=f"[Visualizer] {result['output']}")
        )
        
        return state
    
    def _query_answerer_node(self, state: AgentState) -> AgentState:
        """Query answerer node."""
        query = state["user_query"]
        df = state["dataframe"] or self.df
        
        result = self.query_answerer.answer(df, query)
        
        # Update state
        state["current_agent"] = "query_answerer"
        state["final_response"] = result["output"]
        state["analysis_results"]["query_answerer"] = result
        
        # Add to messages
        state["messages"].append(
            AIMessage(content=f"[Query Answerer] {result['output']}")
        )
        
        return state
    
    def _route_query(self, state: AgentState) -> str:
        """Determine which agent to route to based on supervisor's decision."""
        return state["next_action"]
    
    def invoke(self, query: str) -> dict:
        """
        Execute the workflow with a user query.
        
        Args:
            query: The user's question or request
            
        Returns:
            Dict with final response and metadata
        """
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "dataframe": self.df,
            "dataframe_info": {
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            "current_agent": "",
            "analysis_results": {},
            "visualizations": [],
            "next_action": "",
            "user_query": query,
            "final_response": None
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Extract results
        return {
            "output": final_state.get("final_response", "No response generated"),
            "agent": final_state.get("current_agent", "unknown"),
            "has_plot": len(final_state.get("visualizations", [])) > 0,
            "messages": final_state.get("messages", []),
            "analysis_results": final_state.get("analysis_results", {})
        }


def create_agentic_workflow(df: pd.DataFrame, api_key: str = None) -> AgenticWorkflow:
    """
    Factory function to create an agentic workflow.
    
    Args:
        df: The pandas DataFrame to analyze
        api_key: Google API key (optional)
        
    Returns:
        AgenticWorkflow instance ready to process queries
    """
    Config.validate()
    return AgenticWorkflow(df, api_key)
