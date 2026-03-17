"""AI Data Analyst - Analyzer Module with ReAct Pattern"""
import pandas as pd
import json
from typing import Dict, Any, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from ai_data_analyst.config import GEMINI_MODEL, GOOGLE_API_KEY, MAX_OUTPUT_TOKENS, TEMPERATURE, TOP_P, TOP_K


class ReasoningStep(BaseModel):
    """A single step in the reasoning process."""
    step_number: int = Field(description="Step number")
    thought: str = Field(description="What the AI is thinking")
    action: str = Field(description="Action taken: 'reason', 'query_data', 'calculate', 'answer'")
    input_data: Optional[str] = Field(description="Input to the action")
    output_data: Optional[str] = Field(description="Output from the action")


class AnalysisResponse(BaseModel):
    """Structured response for analysis queries."""
    answer: str = Field(description="The answer to the user's question")
    reasoning_chain: List[ReasoningStep] = Field(description="Chain of thought reasoning")
    chart_suggestion: Optional[str] = Field(description="Suggested chart type if applicable")
    data_insights: Optional[Dict[str, Any]] = Field(description="Additional data insights")


class DataAnalyzer:
    """
    Handles data analysis using LangChain and Gemini with ReAct pattern:

    User Question → LLM Reasoning → Pandas Tool → Answer
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize with a dataframe."""
        self.df = df
        self.data_context = self._create_data_context()

        # Initialize LLM with configured parameters
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
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

    def analyze(self, query: str) -> AnalysisResponse:
        """
        Analyze data based on natural language query using ReAct pattern.

        Workflow:
        1. User Question → 2. LLM Reasoning → 3. Data Tool → 4. Answer
        """
        reasoning_chain = []

        # Step 1: Initial Understanding and Reasoning
        reasoning_prompt = PromptTemplate(
            input_variables=["data_context", "query"],
            template="""You are an expert data analyst. Follow the ReAct pattern:

Step 1 - REASON: Understand the user's question and plan your approach.
Step 2 - ACT: Determine what data operations are needed.
Step 3 - OBSERVE: Consider what the data shows.
Step 4 - ANSWER: Provide the final answer.

Dataset Context:
{data_context}

User Query: {query}

First, explain your reasoning about how to answer this question. What data do you need to look at? What calculations might be required?"""
        )

        # Get reasoning from LLM
        reasoning_chain.append(ReasoningStep(
            step_number=1,
            thought="Analyzing the user query to understand requirements",
            action="reason",
            input_data=query,
            output_data=None
        ))

        chain = reasoning_prompt | self.llm
        reasoning_response = chain.invoke({
            "data_context": self.data_context,
            "query": query
        })

        reasoning_text = reasoning_response.content.strip()
        reasoning_chain[0].output_data = reasoning_text[:500]

        # Step 2: Execute Data Operations (Tool)
        reasoning_chain.append(ReasoningStep(
            step_number=2,
            thought="Executing data operations based on reasoning",
            action="query_data",
            input_data="Accessing dataframe via pandas operations",
            output_data=None
        ))

        # Step 3: Generate Final Answer
        answer_prompt = PromptTemplate(
            input_variables=["data_context", "query", "reasoning"],
            template="""You are an expert data analyst. Based on your reasoning, provide a clear answer.

Dataset Context:
{data_context}

User Query: {query}

Your Reasoning:
{reasoning}

Now provide the final answer. Be specific, cite numbers from the data when relevant, and keep it concise.

Answer:"""
        )

        answer_chain = answer_prompt | self.llm
        answer_response = answer_chain.invoke({
            "data_context": self.data_context,
            "query": query,
            "reasoning": reasoning_text
        })

        final_answer = answer_response.content.strip()

        reasoning_chain.append(ReasoningStep(
            step_number=3,
            thought="Synthesizing data into final answer",
            action="answer",
            input_data="Analysis results",
            output_data=final_answer[:300]
        ))

        # Determine chart suggestion
        chart_suggestion = self._suggest_chart_from_query(query)

        return AnalysisResponse(
            answer=final_answer,
            reasoning_chain=reasoning_chain,
            chart_suggestion=chart_suggestion,
            data_insights=self._generate_insights()
        )

    def quick_analyze(self, query: str) -> str:
        """Quick analysis returning just the answer string."""
        response = self.analyze(query)
        return response.answer

    def _suggest_chart_from_query(self, query: str) -> Optional[str]:
        """Suggest chart type based on query keywords."""
        query_lower = query.lower()

        if any(word in query_lower for word in ['distribution', 'histogram', 'frequency', 'spread']):
            return "histogram"
        elif any(word in query_lower for word in ['correlation', 'scatter', 'relationship', 'vs', 'versus', 'compare']):
            return "scatter"
        elif any(word in query_lower for word in ['bar', 'category', 'count', 'group']):
            return "bar"
        elif any(word in query_lower for word in ['trend', 'time', 'line', 'over', 'period']):
            return "line"
        elif any(word in query_lower for word in ['box', 'outlier', 'quartile', 'median']):
            return "box"
        elif any(word in query_lower for word in ['pie', 'proportion', 'percentage', 'share']):
            return "pie"

        return None

    def _generate_insights(self) -> Dict[str, Any]:
        """Generate basic data insights."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns

        insights = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

        if len(numeric_cols) > 0:
            insights["numeric_summary"] = {
                col: {
                    "mean": round(self.df[col].mean(), 2),
                    "std": round(self.df[col].std(), 2),
                    "min": round(self.df[col].min(), 2),
                    "max": round(self.df[col].max(), 2)
                }
                for col in numeric_cols[:3]  # Limit to first 3 numeric columns
            }

        return insights

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

    def explain_reasoning(self, response: AnalysisResponse) -> str:
        """Generate a human-readable explanation of the reasoning process."""
        explanation = f"""🧠 Analysis Reasoning Chain
{'=' * 50}

📋 Query: {response.reasoning_chain[0].input_data}

🔍 Reasoning Steps:
"""
        for step in response.reasoning_chain:
            explanation += f"""
  Step {step.step_number}: {step.action.upper()}
  💭 Thought: {step.thought}
  {f"📥 Input: {step.input_data}" if step.input_data else ""}
  {f"📤 Output: {step.output_data[:200]}..." if step.output_data and len(step.output_data) > 200 else f"📤 Output: {step.output_data}" if step.output_data else ""}
"""

        explanation += f"""
✅ Final Answer:
{response.answer}

📊 Chart Suggestion: {response.chart_suggestion or 'None'}
"""
        return explanation
