"""AI Data Analyst - Chains Module"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd


class AnalystResponse(BaseModel):
    """Structured response for analyst queries."""
    answer: str = Field(description="The main answer to the query")
    chart_type: Optional[str] = Field(description="Recommended chart type if visualization is needed")
    chart_columns: Optional[List[str]] = Field(description="Columns to use for the chart")
    additional_info: Optional[Dict[str, Any]] = Field(description="Any additional data or statistics")


class AnalystChain:
    """LangChain for data analysis."""

    def __init__(self, df: pd.DataFrame, api_key: str, model: str = "gemini-2.0-flash"):
        """Initialize the chain with dataframe and API key."""
        self.df = df
        self.api_key = api_key
        self.model = model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0,
        )
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,
            prefix="You are an expert data analyst. Use the provided dataframe `df` to answer questions securely and accurately. If calculation is needed, use python to calculate the results."
        )

    def query(self, user_query: str) -> str:
        """Process a natural language query about the data."""
        try:
            response = self.agent.invoke(user_query)
            if isinstance(response, dict) and 'output' in response:
                return str(response['output']).strip()
            return str(response).strip()
        except Exception as e:
            error_msg = str(e)
            if "429 RESOURCE_EXHAUSTED" in error_msg or "Quota exceeded" in error_msg:
                return "⚠️ **Rate Limit Reached**: The free tier of the Gemini API has exhausted its quota (requests per minute/day). Please wait a bit and try again, or upgrade your Gemini API tier."
            return f"Error executing query: {error_msg}"

    def suggest_chart(self, user_query: str) -> Dict[str, Any]:
        """Suggest appropriate chart based on query."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        prompt = PromptTemplate(
            input_variables=["query", "numeric_cols", "categorical_cols"],
            template="""Based on the user's question and available columns, suggest the best chart.

Available Numeric Columns: {numeric_cols}
Available Categorical Columns: {categorical_cols}

User Question: {query}

Respond with just the chart type (histogram, bar, scatter, line, box, pie) and which columns to use.
Format: chart_type|column1,column2"""
        )

        chain = prompt | self.llm

        response = chain.invoke({
            "query": user_query,
            "numeric_cols": str(numeric_cols),
            "categorical_cols": str(cat_cols)
        })

        # Parse the response
        try:
            parts = response.content.strip().split('|')
            chart_type = parts[0].strip().lower()
            columns = [c.strip() for c in parts[1].split(',')] if len(parts) > 1 else []

            return {"chart_type": chart_type, "columns": columns}
        except Exception:
            return {"chart_type": "bar", "columns": [cat_cols[0]] if cat_cols else numeric_cols[:1]}

    def explain_column(self, column: str) -> str:
        """Explain what a column represents."""
        if column not in self.df.columns:
            return f"Column '{column}' not found in the dataset."

        col_data = self.df[column]
        dtype = str(col_data.dtype)

        prompt = PromptTemplate(
            input_variables=["column", "dtype", "sample_values", "stats"],
            template="""Explain what the column '{column}' likely represents based on:
- Data type: {dtype}
- Sample values: {sample_values}
- Basic stats: {stats}

Provide a brief explanation of what this column represents."""
        )

        sample = col_data.dropna().head(5).tolist()
        stats = col_data.describe().to_string() if col_data.dtype in ['int64', 'float64'] else f"Unique values: {col_data.nunique()}"

        chain = prompt | self.llm
        response = chain.invoke({
            "column": column,
            "dtype": dtype,
            "sample_values": str(sample),
            "stats": stats
        })

        return response.content.strip()
