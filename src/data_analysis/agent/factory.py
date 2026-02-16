import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def create_analyst_agent(df: pd.DataFrame, api_key: str = None):
    """
    Creates a LangChain Pandas DataFrame Agent powered by Google Gemini.
    
    Args:
        df: The pandas DataFrame to analyze.
        api_key: Google API Key. If None, checks GOOGLE_API_KEY env var.
        
    Returns:
        A LangChain AgentExecutor ready to run queries.
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        
    if not api_key:
        raise ValueError("Google API Key is required. Please set GOOGLE_API_KEY environment variable or pass it explicitly.")

    # specialized model for data analysis (using flash for speed/cost or pro for reasoning)
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key,
        temperature=0, # Low temperature for code generation
        convert_system_message_to_human=True
    )

    # Prefix to guide the agent
    prefix = """
    You are an expert Data Analyst using Python. EXPERT INSTRUCTION: 
    1. You are given a dataframe 'df'.
    2. When asked to plot, use 'matplotlib.pyplot' as 'plt'.
    3. IMPORTANT: After creating a plot with `plt.plot(...)` or similar, you DO NOT need to show it. The system will capture the current figure automatically.
    4. If the user asks for a specific value or summary, print it or return it as the final answer.
    5. Always deal with missing values gracefully (e.g., drop or fill) if they cause errors.
    """

    return create_pandas_dataframe_agent(
        model,
        df,
        verbose=True,
        agent_type="zero-shot-react-description",
        prefix=prefix,
        allow_dangerous_code=True # Required for executing pandas code
    )
