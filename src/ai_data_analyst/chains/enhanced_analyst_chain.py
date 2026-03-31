"""AI Data Analyst - Enhanced Chains Module with Improved ReAct Pattern

Improvements:
1. Better system prompts with few-shot examples
2. Enhanced reasoning chain with validation steps
3. Query classification for better response routing
4. Confidence scoring for answers
5. Error recovery and fallback strategies
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import re
from ai_data_analyst.config import MAX_OUTPUT_TOKENS, TEMPERATURE, TOP_P, TOP_K


class AnalysisStep(BaseModel):
    """Represents a single step in the analysis workflow with confidence."""
    step: int = Field(description="Step number in the workflow")
    thought: str = Field(description="The reasoning/thought process")
    action: str = Field(description="The action taken (e.g., 'query_df', 'calculate', 'filter')")
    action_input: str = Field(description="The input to the action")
    observation: Optional[str] = Field(description="The result/observation from the action")
    confidence: float = Field(default=1.0, description="Confidence score for this step (0-1)")


class AnalystResponse(BaseModel):
    """Enhanced structured response with confidence and validation."""
    user_question: str = Field(description="The original user question")
    reasoning_steps: List[AnalysisStep] = Field(description="Step-by-step reasoning process")
    final_answer: str = Field(description="The final natural language answer")
    chart_type: Optional[str] = Field(description="Recommended chart type if visualization is needed")
    chart_columns: Optional[List[str]] = Field(description="Columns to use for the chart")
    data_insights: Optional[Dict[str, Any]] = Field(description="Additional data insights discovered")
    confidence_score: float = Field(default=1.0, description="Overall confidence in the answer (0-1)")
    query_type: str = Field(description="Type of query: aggregation, comparison, trend, distribution, correlation")
    validation_notes: Optional[str] = Field(description="Notes about answer validation")


class QueryClassifier:
    """Classifies user queries to route them to appropriate analysis strategies."""

    QUERY_TYPES = {
        'aggregation': ['total', 'sum', 'average', 'mean', 'count', 'min', 'max', 'median'],
        'comparison': ['compare', 'comparison', 'vs', 'versus', 'difference', 'higher', 'lower', 'better', 'worse'],
        'trend': ['trend', 'over time', 'change', 'growth', 'decline', 'increase', 'decrease', 'pattern'],
        'distribution': ['distribution', 'spread', 'frequency', 'histogram', 'range', 'percentile'],
        'correlation': ['correlation', 'relationship', 'associated', 'linked', 'predict', 'influence'],
        'top_n': ['top', 'best', 'highest', 'largest', 'most', 'bottom', 'worst', 'lowest'],
        'filter': ['filter', 'where', 'which', 'only', 'specific', 'particular'],
    }

    @classmethod
    def classify(cls, query: str) -> tuple[str, float]:
        """Classify query and return type with confidence."""
        query_lower = query.lower()
        
        scores = {}
        for query_type, keywords in cls.QUERY_TYPES.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[query_type] = score
        
        if max(scores.values()) == 0:
            return 'general', 0.5
        
        best_type = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_type] / 3.0)  # Normalize to 0-1
        
        return best_type, confidence


class EnhancedAnalystChain:
    """
    Enhanced LangChain Agent with improved ReAct pattern:
    
    Improvements over base AnalystChain:
    1. Query classification for better routing
    2. Few-shot examples in prompts
    3. Multi-step validation
    4. Confidence scoring
    5. Better error handling with recovery
    6. Context-aware follow-up suggestions
    """

    # Few-shot examples for better LLM guidance
    EXAMPLES = [
        {
            "query": "What is the total sales?",
            "reasoning": "This is an aggregation query. I need to sum the 'sales' column.",
            "action": "df['sales'].sum()",
            "answer": "The total sales across all records is ${value:,.2f}."
        },
        {
            "query": "Which category has the highest average profit?",
            "reasoning": "This is a comparison query. I need to group by category and calculate mean profit.",
            "action": "df.groupby('category')['profit'].mean().idxmax()",
            "answer": "The category with the highest average profit is {category} with ${value:,.2f}."
        },
        {
            "query": "Show me the trend of sales over quarters",
            "reasoning": "This is a trend query. I need to group by quarter and track sales changes.",
            "action": "df.groupby('quarter')['sales'].sum()",
            "answer": "Sales trend by quarter: {quarter_data}. The trend shows {trend_description}."
        },
        {
            "query": "What's the correlation between sales and profit?",
            "reasoning": "This is a correlation query. I need to calculate the correlation coefficient.",
            "action": "df['sales'].corr(df['profit'])",
            "answer": "The correlation between sales and profit is {value:.3f}, indicating {strength} relationship."
        },
    ]

    def __init__(self, df: pd.DataFrame, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialize the enhanced chain with dataframe and API key."""
        self.df = df
        self.api_key = api_key
        self.model = model
        self.data_summary = self._create_data_summary()

        # Initialize LLM with configured parameters
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
        )

        # Create enhanced ReAct agent
        self.agent = self._create_enhanced_agent()

    def _create_data_summary(self) -> str:
        """Create a comprehensive data summary for context."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        summary = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'numeric_stats': {},
            'categorical_stats': {},
        }
        
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            summary['numeric_stats'][col] = {
                'mean': f"{self.df[col].mean():.2f}",
                'std': f"{self.df[col].std():.2f}",
                'min': f"{self.df[col].min():.2f}",
                'max': f"{self.df[col].max():.2f}",
            }
        
        for col in cat_cols[:5]:  # Limit to first 5 categorical columns
            summary['categorical_stats'][col] = {
                'unique': self.df[col].nunique(),
                'top': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A',
            }
        
        return summary

    def _create_enhanced_agent(self):
        """Create enhanced ReAct agent with better system prompt."""
        system_prompt = f"""You are an expert data analyst AI assistant. Follow the Enhanced ReAct workflow:

WORKFLOW:
1. CLASSIFY: Identify the query type (aggregation, comparison, trend, distribution, correlation, top_n, filter)
2. UNDERSTAND: Analyze what the user needs and what data columns are relevant
3. REASON: Think step-by-step about the operations needed
4. ACT: Execute pandas operations using the python_repl_ast tool
5. VALIDATE: Check if the result makes sense (no NaN, reasonable values)
6. ANSWER: Provide a clear, specific answer with numbers and context

DATA CONTEXT:
- Dataset shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns
- Columns: {list(self.df.columns)}
- Numeric columns: {list(self.df.select_dtypes(include=['number']).columns)}
- Categorical columns: {list(self.df.select_dtypes(include=['object', 'category']).columns)}

RULES:
- ALWAYS think before acting - explain your reasoning
- Use df.head(), df.info(), df.describe() to explore if unsure
- For aggregations, specify the column and operation clearly
- For comparisons, identify the grouping column and metric
- Validate results: check for NaN, infinite values, or unreasonable numbers
- Provide specific numbers with appropriate units (%, $, etc.)
- If uncertain, acknowledge limitations and provide best estimate
- For trend questions, mention direction and magnitude of change
- For correlations, interpret the strength (weak/moderate/strong)

EXAMPLES:
Q: "What is the total sales?"
A: "I need to sum the sales column. Using df['sales'].sum(), the total is $X."

Q: "Which region has highest profit?"
A: "I'll group by region and find max average profit. Result: Region X with $Y."

Available tool: python_repl_ast - Execute pandas code in `df` variable
"""

        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="zero-shot-react-description",
            prefix=system_prompt,
        )
        
        return self.agent

    def analyze(self, user_query: str) -> AnalystResponse:
        """
        Execute enhanced ReAct analysis with classification and validation.
        """
        # Step 1: Classify the query
        query_type, classification_confidence = QueryClassifier.classify(user_query)
        
        reasoning_steps = []
        
        # Step 2: Add classification step
        reasoning_steps.append(AnalysisStep(
            step=1,
            thought=f"Classifying query type to determine analysis strategy",
            action="classify",
            action_input=user_query,
            observation=f"Query type: {query_type} (confidence: {classification_confidence:.2f})",
            confidence=classification_confidence
        ))

        # Step 3: Execute ReAct agent with enhanced prompt
        try:
            agent_response = self.agent.invoke(user_query)
            
            # Extract output and intermediate steps
            if isinstance(agent_response, dict):
                final_output = str(agent_response.get('output', str(agent_response))).strip()
                
                # Capture intermediate steps if available
                intermediate_steps = agent_response.get('intermediate_steps', [])
                for i, step in enumerate(intermediate_steps, start=2):
                    if isinstance(step, tuple) and len(step) == 2:
                        action, observation = step
                        reasoning_steps.append(AnalysisStep(
                            step=i,
                            thought="Executing data operation based on reasoning",
                            action=str(getattr(action, 'tool', 'unknown')),
                            action_input=str(getattr(action, 'tool_input', '')),
                            observation=str(observation)[:500],
                            confidence=0.9
                        ))
            else:
                final_output = str(agent_response).strip()
                
            # Step 4: Validate the answer
            validation_result = self._validate_answer(final_output, query_type)
            
            if validation_result['is_valid']:
                validation_confidence = validation_result['confidence']
                validation_notes = validation_result['notes']
            else:
                validation_confidence = 0.5
                validation_notes = f"Validation warning: {validation_result['notes']}"
                final_output += f"\n\n⚠️ Note: {validation_result['notes']}"
            
            reasoning_steps.append(AnalysisStep(
                step=len(reasoning_steps) + 1,
                thought="Validating the answer for accuracy and reasonableness",
                action="validate",
                action_input=final_output[:200],
                observation=validation_notes,
                confidence=validation_confidence
            ))
            
            # Calculate overall confidence
            overall_confidence = self._calculate_confidence(
                classification_confidence,
                validation_confidence,
                len(reasoning_steps)
            )
            
        except Exception as e:
            error_msg = str(e)
            final_output = self._handle_error(error_msg, user_query)
            overall_confidence = 0.3
            validation_notes = f"Error occurred: {error_msg[:100]}"
            
            reasoning_steps.append(AnalysisStep(
                step=len(reasoning_steps) + 1,
                thought="Error during analysis execution",
                action="error",
                action_input=user_query,
                observation=error_msg,
                confidence=0.3
            ))

        # Determine chart suggestion
        chart_info = self._suggest_chart(user_query, query_type)

        return AnalystResponse(
            user_question=user_query,
            reasoning_steps=reasoning_steps,
            final_answer=final_output,
            chart_type=chart_info.get("chart_type"),
            chart_columns=chart_info.get("columns"),
            data_insights=self._extract_insights(),
            confidence_score=overall_confidence,
            query_type=query_type,
            validation_notes=validation_notes
        )

    def _validate_answer(self, answer: str, query_type: str) -> Dict[str, Any]:
        """Validate the answer for common issues."""
        issues = []
        confidence = 1.0
        
        # Check for NaN or None
        if 'nan' in answer.lower() or 'none' in answer.lower():
            issues.append("Result contains NaN or None values")
            confidence -= 0.3
        
        # Check for error messages
        if 'error' in answer.lower() or 'exception' in answer.lower():
            issues.append("Result contains error messages")
            confidence -= 0.4
        
        # Check for reasonable numbers (if applicable)
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer)
        if numbers and query_type in ['aggregation', 'comparison', 'top_n']:
            try:
                nums = [float(n) for n in numbers]
                # Check for extremely large or small values
                if any(abs(n) > 1e12 for n in nums):
                    issues.append("Result contains extremely large values")
                    confidence -= 0.2
            except:
                pass
        
        return {
            'is_valid': len(issues) == 0,
            'confidence': max(0.1, confidence),
            'notes': "; ".join(issues) if issues else "Answer validated successfully"
        }

    def _calculate_confidence(self, classification_conf: float, validation_conf: float, steps: int) -> float:
        """Calculate overall confidence score."""
        # Weight factors
        base_confidence = (classification_conf * 0.3 + validation_conf * 0.7)
        
        # Bonus for thorough reasoning (3-5 steps is ideal)
        if 3 <= steps <= 5:
            reasoning_bonus = 0.1
        elif steps > 5:
            reasoning_bonus = 0.05
        else:
            reasoning_bonus = 0
        
        return min(1.0, base_confidence + reasoning_bonus)

    def _handle_error(self, error_msg: str, user_query: str) -> str:
        """Handle errors with helpful fallback responses."""
        if "429 RESOURCE_EXHAUSTED" in error_msg or "Quota exceeded" in error_msg:
            return "⚠️ **Rate Limit Reached**: The Gemini API quota has been exceeded. Please wait a moment and try again."
        
        if "column" in error_msg.lower() and "not found" in error_msg.lower():
            return f"⚠️ **Column Error**: I couldn't find the specified column. Available columns are: {', '.join(self.df.columns)}"
        
        if "dtype" in error_msg.lower() or "type" in error_msg.lower():
            return f"⚠️ **Data Type Error**: The operation requires different data types. Please check if you're using numeric columns for calculations."
        
        return f"⚠️ **Analysis Error**: I encountered an issue while analyzing your query. Please try rephrasing: '{user_query}'"

    def _suggest_chart(self, user_query: str, query_type: str) -> Dict[str, Any]:
        """Enhanced chart suggestion based on query type and data."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        chart_mapping = {
            'aggregation': ('bar', cat_cols[:1] + numeric_cols[:1] if cat_cols and numeric_cols else numeric_cols[:1]),
            'comparison': ('bar', cat_cols[:1] + numeric_cols[:1] if cat_cols and numeric_cols else []),
            'trend': ('line', [cat_cols[0]] + numeric_cols[:1] if cat_cols and numeric_cols else numeric_cols[:2]),
            'distribution': ('histogram', numeric_cols[:1] if numeric_cols else []),
            'correlation': ('scatter', numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols),
            'top_n': ('bar', cat_cols[:1] + numeric_cols[:1] if cat_cols and numeric_cols else []),
            'filter': ('pie', cat_cols[:1] if cat_cols else []),
            'general': ('bar', numeric_cols[:1] if numeric_cols else []),
        }
        
        chart_type, columns = chart_mapping.get(query_type, ('bar', []))
        
        return {
            "chart_type": chart_type,
            "columns": [c for c in columns if c in self.df.columns]
        }

    def _extract_insights(self) -> Dict[str, Any]:
        """Extract enhanced data insights."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        insights = {
            "dataset_shape": f"{self.df.shape[0]} rows × {self.df.shape[1]} columns",
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(self.df.select_dtypes(include=['object', 'category']).columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
        }
        
        if len(numeric_cols) > 0:
            # Add top correlations
            if len(numeric_cols) >= 2:
                corr_matrix = self.df[numeric_cols].corr()
                # Find highest correlation (excluding self-correlation)
                max_corr = 0
                corr_pair = None
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:
                            corr_val = abs(corr_matrix.loc[col1, col2])
                            if corr_val > max_corr:
                                max_corr = corr_val
                                corr_pair = (col1, col2)
                
                if corr_pair:
                    insights["strongest_correlation"] = {
                        "columns": list(corr_pair),
                        "coefficient": round(max_corr, 3)
                    }
        
        return insights

    def query(self, user_query: str) -> str:
        """Simple query method - returns just the answer."""
        response = self.analyze(user_query)
        return response.final_answer

    def get_workflow_summary(self, response: AnalystResponse) -> str:
        """Generate enhanced human-readable workflow summary."""
        confidence_emoji = "✅" if response.confidence_score >= 0.8 else "⚠️" if response.confidence_score >= 0.5 else "❌"
        
        summary = f"""📊 Enhanced Analysis Workflow {confidence_emoji}
{'=' * 50}

📝 User Question: {response.user_question}
🏷️ Query Type: {response.query_type}
📈 Confidence: {response.confidence_score:.1%}

🔍 Reasoning Steps:
"""
        for step in response.reasoning_steps:
            step_confidence = "✅" if step.confidence >= 0.8 else "⚠️" if step.confidence >= 0.5 else "❌"
            summary += f"""
  {step_confidence} Step {step.step}: {step.action.upper()}
     Thought: {step.thought}
     {f"Observation: {step.observation}" if step.observation else ""}
"""

        summary += f"""
✅ Final Answer: {response.final_answer}

📈 Suggested Visualization: {response.chart_type or 'None'}
   Columns: {', '.join(response.chart_columns) if response.chart_columns else 'N/A'}

📋 Validation: {response.validation_notes}
"""
        return summary

    def suggest_follow_up_questions(self, response: AnalystResponse) -> List[str]:
        """Generate context-aware follow-up question suggestions."""
        suggestions = []
        
        query_type = response.query_type
        chart_cols = response.chart_columns or []
        
        if query_type == 'aggregation':
            suggestions.append(f"Can you break this down by category?")
            suggestions.append(f"How does this compare to the average?")
            suggestions.append(f"What's the trend over time?")
        
        elif query_type == 'comparison':
            suggestions.append(f"Can you show this as a visualization?")
            suggestions.append(f"What factors contribute to this difference?")
            suggestions.append(f"Is this difference statistically significant?")
        
        elif query_type == 'trend':
            suggestions.append(f"What's driving this trend?")
            suggestions.append(f"Can you forecast the next period?")
            suggestions.append(f"Are there any seasonal patterns?")
        
        elif query_type == 'correlation':
            suggestions.append(f"Can you visualize this relationship?")
            suggestions.append(f"Does correlation imply causation here?")
            suggestions.append(f"What other variables are correlated?")
        
        elif query_type == 'top_n':
            suggestions.append(f"What characteristics do the top performers share?")
            suggestions.append(f"How do the bottom performers compare?")
            suggestions.append(f"Can you show the distribution?")
        
        # Add data-specific suggestions
        if chart_cols and len(chart_cols) >= 2:
            suggestions.append(f"Can you analyze {chart_cols[0]} vs {chart_cols[1]}?")
        
        return suggestions[:3]  # Return top 3 suggestions
