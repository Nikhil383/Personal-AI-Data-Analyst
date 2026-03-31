"""AI Data Analyst - Chains Module with ReAct Pattern"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
from ai_data_analyst.config import MAX_OUTPUT_TOKENS, TEMPERATURE, TOP_P, TOP_K


class AnalysisStep(BaseModel):
    """Represents a single step in the analysis workflow."""
    step: int = Field(description="Step number in the workflow")
    thought: str = Field(description="The reasoning/thought process")
    action: str = Field(description="The action taken (e.g., 'query_df', 'calculate', 'filter')")
    action_input: str = Field(description="The input to the action")
    observation: Optional[str] = Field(description="The result/observation from the action")


class AnalystResponse(BaseModel):
    """Structured response for analyst queries with full workflow."""
    user_question: str = Field(description="The original user question")
    reasoning_steps: List[AnalysisStep] = Field(description="Step-by-step reasoning process")
    final_answer: str = Field(description="The final natural language answer")
    chart_type: Optional[str] = Field(description="Recommended chart type if visualization is needed")
    chart_columns: Optional[List[str]] = Field(description="Columns to use for the chart")
    data_insights: Optional[Dict[str, Any]] = Field(description="Additional data insights discovered")


class AnalystChain:
    """
    LangChain Agent for data analysis following ReAct pattern:

    User Question → LangChain Agent → Gemini LLM (reasoning) → Tool (Pandas) → Answer
    """

    def __init__(self, df: pd.DataFrame, api_key: str, model: str = "gemini-2.0-flash"):
        """Initialize the chain with dataframe and API key."""
        self.df = df
        self.api_key = api_key
        self.model = model

        # Initialize LLM with configured parameters
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
        )

        # Create the ReAct agent with enhanced system prompt
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="zero-shot-react-description",
            prefix="""You are an expert data analyst AI. Follow this ReAct workflow:

WORKFLOW:
1. UNDERSTAND: Analyze the user's question and understand what they need
2. REASON: Think step-by-step about what data operations are needed
3. ACT: Use the available tools (python_repl_ast) to execute pandas operations
4. OBSERVE: Review the results from the tool execution
5. ANSWER: Provide a clear, natural language answer based on the observations

RULES:
- Always think before acting
- Use df.info(), df.describe(), or df.head() to explore if needed
- Perform calculations using pandas/python when necessary
- Provide specific numbers and insights from the data
- If the answer involves multiple steps, explain your reasoning
- Keep responses concise but informative

Available dataframe: `df` with columns: """ + str(list(df.columns))
        )

    def analyze(self, user_query: str) -> AnalystResponse:
        """
        Execute the full ReAct analysis workflow.

        Flow: User Question → Agent → LLM Reasoning → Pandas Tool → Answer
        """
        steps = []

        # Step 1: Initial Understanding
        steps.append(AnalysisStep(
            step=1,
            thought=f"Received user query: '{user_query}'. Need to understand what data operations are required.",
            action="understand",
            action_input=user_query,
            observation=None
        ))

        # Step 2-4: Execute ReAct agent (Reasoning → Action → Observation)
        try:
            agent_response = self.agent.invoke(user_query)

            # Extract the final output
            if isinstance(agent_response, dict) and 'output' in agent_response:
                final_output = str(agent_response['output']).strip()
                # Capture intermediate steps if available
                if 'intermediate_steps' in agent_response:
                    for i, step in enumerate(agent_response['intermediate_steps'], start=2):
                        action, observation = step if isinstance(step, tuple) else (step, "")
                        steps.append(AnalysisStep(
                            step=i,
                            thought=f"Executing tool to gather data...",
                            action=str(action),
                            action_input=str(action),
                            observation=str(observation)[:500]  # Truncate long observations
                        ))
            else:
                final_output = str(agent_response).strip()

        except Exception as e:
            error_msg = str(e)
            if "429 RESOURCE_EXHAUSTED" in error_msg or "Quota exceeded" in error_msg:
                final_output = "⚠️ **Rate Limit Reached**: The free tier of the Gemini API has exhausted its quota. Please wait and try again."
            else:
                final_output = f"Error executing query: {error_msg}"

            steps.append(AnalysisStep(
                step=2,
                thought="Error occurred during execution",
                action="error",
                action_input=user_query,
                observation=error_msg
            ))

        # Final Step: Formulate Answer
        steps.append(AnalysisStep(
            step=len(steps) + 1,
            thought="Synthesizing results into natural language answer",
            action="generate_answer",
            action_input="final_results",
            observation=final_output[:200]
        ))

        # Determine if chart is suggested
        chart_info = self._suggest_chart(user_query)

        return AnalystResponse(
            user_question=user_query,
            reasoning_steps=steps,
            final_answer=final_output,
            chart_type=chart_info.get("chart_type"),
            chart_columns=chart_info.get("columns"),
            data_insights=self._extract_insights()
        )

    def query(self, user_query: str) -> str:
        """Simple query method - returns just the answer."""
        response = self.analyze(user_query)
        return response.final_answer

    def _suggest_chart(self, user_query: str) -> Dict[str, Any]:
        """Internal method to suggest appropriate chart based on query."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        query_lower = user_query.lower()

        # Simple heuristic-based chart suggestion
        if any(word in query_lower for word in ['distribution', 'histogram', 'frequency']):
            return {"chart_type": "histogram", "columns": numeric_cols[:1] if numeric_cols else []}
        elif any(word in query_lower for word in ['correlation', 'scatter', 'relationship', 'vs', 'versus']):
            return {"chart_type": "scatter", "columns": numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols}
        elif any(word in query_lower for word in ['compare', 'comparison', 'bar', 'category']):
            return {"chart_type": "bar", "columns": [cat_cols[0], numeric_cols[0]] if cat_cols and numeric_cols else []}
        elif any(word in query_lower for word in ['trend', 'time', 'line', 'over']):
            return {"chart_type": "line", "columns": numeric_cols[:1] if numeric_cols else []}
        elif any(word in query_lower for word in ['box', 'outlier', 'quartile']):
            return {"chart_type": "box", "columns": numeric_cols[:1] if numeric_cols else []}

        # Default suggestion
        if numeric_cols and cat_cols:
            return {"chart_type": "bar", "columns": [cat_cols[0], numeric_cols[0]]}
        elif len(numeric_cols) >= 2:
            return {"chart_type": "scatter", "columns": numeric_cols[:2]}
        elif numeric_cols:
            return {"chart_type": "histogram", "columns": numeric_cols[:1]}

        return {"chart_type": None, "columns": []}

    def _extract_insights(self) -> Dict[str, Any]:
        """Extract basic data insights for the response."""
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "numeric_columns": len(self.df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(self.df.select_dtypes(include=['object', 'category']).columns),
        }

    def explain_column(self, column: str) -> str:
        """Explain what a column represents using the ReAct pattern."""
        if column not in self.df.columns:
            return f"Column '{column}' not found in the dataset."

        col_data = self.df[column]
        dtype = str(col_data.dtype)

        prompt = PromptTemplate(
            input_variables=["column", "dtype", "sample_values", "stats"],
            template="""You are a data analyst. Explain what the column '{column}' likely represents.

Data type: {dtype}
Sample values: {sample_values}
Basic stats: {stats}

Follow this reasoning:
1. Look at the column name and data type
2. Examine the sample values
3. Consider the statistical distribution
4. Provide a brief explanation of what this column represents

Explanation:"""
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

    def get_workflow_summary(self, response: AnalystResponse) -> str:
        """Generate a human-readable summary of the analysis workflow."""
        summary = f"""📊 Analysis Workflow
{'=' * 50}

📝 User Question: {response.user_question}

🔍 Reasoning Steps:
"""
        for step in response.reasoning_steps:
            summary += f"""
  Step {step.step}: {step.action.upper()}
  Thought: {step.thought}
  {f"Observation: {step.observation}" if step.observation else ""}
"""

        summary += f"""
✅ Final Answer: {response.final_answer}

📈 Suggested Chart: {response.chart_type or 'None'}
   Columns: {', '.join(response.chart_columns) if response.chart_columns else 'N/A'}
"""
        return summary
