"""AI Data Analyst - Chains Module"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
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
            temperature=0.3,
        )

    def _build_data_summary(self) -> str:
        """Build a summary of the data for context."""
        summary_parts = [
            f"Dataset has {len(self.df)} rows and {len(self.df.columns)} columns.",
            f"Columns: {', '.join(self.df.columns)}.",
        ]

        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            summary_parts.append(f"Numeric columns: {', '.join(numeric_cols)}.")

        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            summary_parts.append(f"Categorical columns: {', '.join(cat_cols)}.")

        return " ".join(summary_parts)

    def query(self, user_query: str) -> str:
        """Process a natural language query about the data."""
        data_summary = self._build_data_summary()

        prompt = PromptTemplate(
            input_variables=["data_summary", "query", "data_preview"],
            template="""You are an expert data analyst. Use the provided data to answer questions.

Data Summary: {data_summary}

Data Preview (first 10 rows):
{data_preview}

User Question: {query}

Provide a clear, accurate answer based on the data. If calculation is needed, show the results."""
        )

        chain = prompt | self.llm
        data_preview = self.df.head(10).to_string()

        response = chain.invoke({
            "data_summary": data_summary,
            "query": user_query,
            "data_preview": data_preview
        })

        return response.content.strip()

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
