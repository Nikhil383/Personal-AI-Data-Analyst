"""
Data Cleaner Agent - Handles data cleaning operations.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.messages import SystemMessage
import pandas as pd

from ..config import Config


class DataCleanerAgent:
    """
    Specialized agent for data cleaning tasks.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the data cleaner agent."""
        api_key = api_key or Config.GOOGLE_API_KEY
        
        self.llm = ChatGoogleGenerativeAI(
            model=Config.AGENT_MODEL,
            google_api_key=api_key,
            temperature=Config.AGENT_TEMPERATURE,
        )
        
        self.system_prompt = """You are a data cleaning specialist.

Your expertise includes:
- Handling missing values (drop, fill with mean/median/mode, forward/backward fill, interpolation)
- Removing duplicate rows
- Detecting and handling outliers
- Fixing data types (convert strings to numbers, dates, etc.)
- Standardizing column names
- Removing unnecessary whitespace

When cleaning data:
1. Always explain what you're doing and why
2. Report statistics before and after cleaning (e.g., "Removed 15 duplicates, filled 23 missing values")
3. Be conservative - don't delete data unless necessary
4. Suggest the best approach but explain alternatives

IMPORTANT: You have access to a pandas DataFrame called 'df'. Use pandas operations to clean the data.
After cleaning, return a summary of what was done."""
    
    def create_agent(self, df: pd.DataFrame):
        """
        Create a pandas dataframe agent for this specific dataframe.
        
        Args:
            df: The dataframe to clean
            
        Returns:
            LangChain agent executor
        """
        return create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,
            agent_type="zero-shot-react-description",
            prefix=self.system_prompt,
            allow_dangerous_code=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
    
    def clean(self, df: pd.DataFrame, query: str) -> dict:
        """
        Execute a cleaning operation on the dataframe.
        
        Args:
            df: The dataframe to clean
            query: The cleaning task description
            
        Returns:
            Dict with 'output' (explanation) and 'dataframe' (cleaned df)
        """
        agent = self.create_agent(df)
        
        try:
            response = agent.invoke(query)
            output_text = response.get("output", str(response))
            
            return {
                "output": output_text,
                "dataframe": df,  # The df is modified in place
                "agent": "data_cleaner"
            }
        except Exception as e:
            return {
                "output": f"Error during cleaning: {str(e)}",
                "dataframe": df,
                "agent": "data_cleaner",
                "error": True
            }
