"""
Query Answerer Agent - Answers general questions about data.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

from ..config import Config


class QueryAnswererAgent:
    """
    Specialized agent for answering general questions about the dataset.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the query answerer agent."""
        api_key = api_key or Config.GOOGLE_API_KEY
        
        self.llm = ChatGoogleGenerativeAI(
            model=Config.AGENT_MODEL,
            google_api_key=api_key,
            temperature=Config.AGENT_TEMPERATURE,
        )
        
        self.system_prompt = """You are a data analysis expert who answers questions about datasets.

Your capabilities include:
- Describing the structure and contents of the data
- Answering specific questions about values, rows, columns
- Providing insights and summaries
- Explaining patterns and trends
- Generating reports
- Comparing different segments of data

When answering questions:
1. Be precise and accurate
2. Use actual data from the dataframe to support your answers
3. Format your responses clearly (use bullet points, tables when appropriate)
4. If you're not certain, say so
5. Provide context and explanations, not just numbers

IMPORTANT: You have access to a pandas DataFrame called 'df'. Use pandas operations to query the data.
You can use methods like .head(), .tail(), .info(), .value_counts(), filtering, groupby, etc.

If the user asks for formatted output, you can use the 'tabulate' library:
```python
from tabulate import tabulate
print(tabulate(df.head(), headers='keys', tablefmt='grid'))
```

Provide comprehensive, helpful answers."""
    
    def create_agent(self, df: pd.DataFrame):
        """
        Create a pandas dataframe agent for this specific dataframe.
        
        Args:
            df: The dataframe to query
            
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
    
    def answer(self, df: pd.DataFrame, query: str) -> dict:
        """
        Answer a question about the dataframe.
        
        Args:
            df: The dataframe to query
            query: The question to answer
            
        Returns:
            Dict with 'output' (answer) and metadata
        """
        agent = self.create_agent(df)
        
        try:
            response = agent.invoke(query)
            output_text = response.get("output", str(response))
            
            return {
                "output": output_text,
                "agent": "query_answerer"
            }
        except Exception as e:
            return {
                "output": f"Error answering query: {str(e)}",
                "agent": "query_answerer",
                "error": True
            }
