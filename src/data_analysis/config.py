"""
Configuration for the agentic AI data analyst system.
"""
import os
from typing import Optional
from .utils import APIKeyValidator


class Config:
    """Central configuration for the multi-agent system."""
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    
    # Model configurations
    SUPERVISOR_MODEL = "gemini-1.5-pro"  # More powerful for routing decisions
    AGENT_MODEL = "gemini-1.5-flash"  # Faster for specialized tasks
    
    # Temperature settings
    SUPERVISOR_TEMPERATURE = 0.1
    AGENT_TEMPERATURE = 0
    
    # Token limits
    MAX_TOKENS = 8192
    
    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    
    # LangSmith (observability)
    ENABLE_LANGSMITH = bool(LANGSMITH_API_KEY)
    LANGSMITH_PROJECT = "ai-data-analyst"
    
    @classmethod
    def validate(cls):
        """Validate required configuration using APIKeyValidator."""
        APIKeyValidator.raise_if_missing_required()

