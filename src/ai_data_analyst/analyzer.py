"""AI Data Analyst - Analyzer Module"""
import pandas as pd
import json
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config import GEMINI_MODEL, GOOGLE_API_KEY


class AnalysisResponse(BaseModel):
    """Structured response for analysis queries."""
    answer: str = Field(description="The answer to the user's question")
    chart_suggestion: Optional[str] = Field(description="Suggested chart type if applicable")
    data_insights: Optional[Dict[str, Any]] = Field(description="Additional data insights")


class DataAnalyzer:
    """Handles data analysis using LangChain and Gemini."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a dataframe."""
        self.df = df
        self.data_context = self._create_data_context()
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
        )

    def _create_data_context(self) -> str:
        """Create a context string describing the data."""
        info = {
            'rows': len(self.df),
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'numeric_cols': list(self.df.select_dtypes(include=['number']).columns),
            'categorical_cols': list(self.df.select_dtypes(include=['object', 'category']).columns),
            'preview': self.df.head(10).to_string(),
            'describe': self.df.describe().to_string(),
        }

        context = f"""
Dataset Information:
- Total Rows: {info['rows']}
- Total Columns: {info['columns']}
- Column Names: {', '.join(info['columns'])}
- Numeric Columns: {', '.join(info['numeric_cols']) if info['numeric_cols'] else 'None'}
- Categorical Columns: {', '.join(info['categorical_cols']) if info['categorical_cols'] else 'None'}

Data Types:
{json.dumps(info['dtypes'], indent=2)}

First 10 Rows:
{info['preview']}

Summary Statistics:
{info['describe']}
"""
        return context

    def analyze(self, query: str) -> str:
        """Analyze data based on natural language query."""
        prompt_template = PromptTemplate(
            input_variables=["data_context", "query"],
            template="""You are an expert data analyst. You have access to a dataset and need to answer questions about it.

Dataset Context:
{data_context}

User Query: {query}

Instructions:
1. Analyze the data to answer the user's question
2. Provide clear, concise answers based on the data
3. If the query asks for calculations, show the results
4. If applicable, suggest what type of chart would visualize this data well
5. Keep your response focused and actionable

Answer:"""
        )

        chain = prompt_template | self.llm
        response = chain.invoke({"data_context": self.data_context, "query": query})
        return response.content.strip()

    def get_column_analysis(self, column: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific column."""
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found"}

        col_data = self.df[column]
        dtype = str(col_data.dtype)

        analysis = {
            "column": column,
            "dtype": dtype,
            "null_count": int(col_data.isnull().sum()),
            "unique_count": int(col_data.nunique()),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            analysis.update({
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q25": float(col_data.quantile(0.25)),
                "q75": float(col_data.quantile(0.75)),
            })
        else:
            # For categorical data
            value_counts = col_data.value_counts().head(10)
            analysis["top_values"] = {str(k): int(v) for k, v in value_counts.items()}

        return analysis

    def suggest_visualizations(self) -> list:
        """Suggest appropriate visualizations for the data."""
        suggestions = []
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(numeric_cols) >= 1:
            suggestions.append({
                "type": "histogram",
                "description": "Distribution of numeric columns",
                "columns": numeric_cols[:3]
            })

        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "scatter",
                "description": "Correlation between numeric columns",
                "columns": numeric_cols[:2]
            })

        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append({
                "type": "bar",
                "description": "Average of numeric by category",
                "columns": [categorical_cols[0], numeric_cols[0]]
            })

        if len(numeric_cols) >= 1:
            suggestions.append({
                "type": "box",
                "description": "Box plot of numeric columns",
                "columns": numeric_cols[:3]
            })

        return suggestions

    def filter_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """Filter data based on conditions."""
        try:
            df_filtered = self.df.copy()
            for column, value in conditions.items():
                if column in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered[column] == value]
            return df_filtered
        except Exception as e:
            return self.df

    def aggregate_data(self, group_by: str, agg_col: str, agg_func: str = 'mean') -> pd.DataFrame:
        """Aggregate data by a column."""
        if group_by not in self.df.columns or agg_col not in self.df.columns:
            return pd.DataFrame()

        try:
            agg_funcs = {
                'mean': 'mean',
                'sum': 'sum',
                'count': 'count',
                'min': 'min',
                'max': 'max',
            }
            func = agg_funcs.get(agg_func, 'mean')
            result = self.df.groupby(group_by)[agg_col].agg(func).reset_index()
            return result
        except Exception:
            return pd.DataFrame()
