"""
Statistical Analysis Agent - Performs statistical analysis.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

from ..config import Config


class StatsAnalystAgent:
    """
    Specialized agent for statistical analysis.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the stats analyst agent."""
        api_key = api_key or Config.GOOGLE_API_KEY
        
        self.llm = ChatGoogleGenerativeAI(
            model=Config.AGENT_MODEL,
            google_api_key=api_key,
            temperature=Config.AGENT_TEMPERATURE,
        )
        
        self.system_prompt = """You are a statistical analysis expert.

Your expertise includes:
- Descriptive statistics (mean, median, mode, std dev, quartiles)
- Correlation analysis (Pearson, Spearman)
- Hypothesis testing (t-tests, chi-square, ANOVA)
- Distribution analysis (normality tests, skewness, kurtosis)
- Trend detection and time series basics
- Outlier detection using statistical methods

When performing analysis:
1. Always explain the statistical concepts in simple terms
2. Report both the numbers AND what they mean
3. Use appropriate statistical tests for the data type
4. Mention assumptions and limitations
5. Format results clearly using tables when helpful

IMPORTANT: You have access to a pandas DataFrame called 'df'. Use pandas and scipy for analysis.
You can use methods like .corr(), .describe(), scipy.stats functions, etc.

Format your output clearly with headers and explanations."""
    
    def create_agent(self, df: pd.DataFrame):
        """
        Create a pandas dataframe agent for this specific dataframe.
        
        Args:
            df: The dataframe to analyze
            
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
            max_iterations=15,
            early_stopping_method="generate"
        )
    
    def analyze(self, df: pd.DataFrame, query: str) -> dict:
        """
        Execute a statistical analysis on the dataframe.
        
        Args:
            df: The dataframe to analyze
            query: The analysis task description
            
        Returns:
            Dict with 'output' (analysis results) and metadata
        """
        agent = self.create_agent(df)
        
        try:
            response = agent.invoke(query)
            output_text = response.get("output", str(response))
            
            return {
                "output": output_text,
                "agent": "stats_analyst"
            }
        except Exception as e:
            return {
                "output": f"Error during analysis: {str(e)}",
                "agent": "stats_analyst",
                "error": True
            }
