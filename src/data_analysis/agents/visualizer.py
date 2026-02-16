"""
Visualizer Agent - Creates charts and visualizations.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import matplotlib.pyplot as plt

from ..config import Config


class VisualizerAgent:
    """
    Specialized agent for creating data visualizations.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the visualizer agent."""
        api_key = api_key or Config.GOOGLE_API_KEY
        
        self.llm = ChatGoogleGenerativeAI(
            model=Config.AGENT_MODEL,
            google_api_key=api_key,
            temperature=Config.AGENT_TEMPERATURE,
        )
        
        self.system_prompt = """You are a data visualization expert.

Your expertise includes creating:
- Line plots (for trends over time)
- Bar charts (for categorical comparisons)
- Scatter plots (for relationships between variables)
- Histograms (for distributions)
- Box plots (for outlier detection and quartiles)
- Heatmaps (for correlation matrices)
- Pie charts (for proportions)

When creating visualizations:
1. Choose the most appropriate chart type for the data
2. Use clear titles and axis labels
3. Add legends when needed
4. Use appropriate colors and styling
5. Explain what the visualization shows

IMPORTANT: You have access to a pandas DataFrame called 'df' and matplotlib.pyplot as 'plt'.

CRITICAL VISUALIZATION INSTRUCTIONS:
- After creating a plot with plt.plot(), plt.bar(), plt.scatter(), etc., DO NOT call plt.show()
- The system will automatically capture the current figure
- Use plt.title(), plt.xlabel(), plt.ylabel() to add labels
- Use plt.legend() if you have multiple series
- For multiple subplots, use plt.subplot() or plt.subplots()

Example:
```python
plt.figure(figsize=(10, 6))
df['column'].plot(kind='bar')
plt.title('My Chart Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
# DO NOT call plt.show() - the system captures automatically
```

Describe what the visualization shows after creating it."""
    
    def create_agent(self, df: pd.DataFrame):
        """
        Create a pandas dataframe agent for this specific dataframe.
        
        Args:
            df: The dataframe to visualize
            
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
    
    def visualize(self, df: pd.DataFrame, query: str) -> dict:
        """
        Create a visualization based on the query.
        
        Args:
            df: The dataframe to visualize
            query: The visualization request
            
        Returns:
            Dict with 'output' (description) and visualization metadata
        """
        agent = self.create_agent(df)
        
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        try:
            response = agent.invoke(query)
            output_text = response.get("output", str(response))
            
            # Check if a plot was created
            has_plot = bool(plt.gcf().get_axes())
            
            return {
                "output": output_text,
                "agent": "visualizer",
                "has_plot": has_plot
            }
        except Exception as e:
            return {
                "output": f"Error during visualization: {str(e)}",
                "agent": "visualizer",
                "error": True,
                "has_plot": False
            }
