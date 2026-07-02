import json
from typing import Dict, Any, List, TypedDict, Optional
from sqlalchemy import text
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from app.config import GOOGLE_API_KEY, GEMINI_MODEL
from app.database.models import engine

# Define state structure
class AgentState(TypedDict):
    query: str
    intent: str
    sql_query: Optional[str]
    sql_valid: Optional[bool]
    sql_error: Optional[str]
    query_results: Optional[List[Dict[str, Any]]]
    visualization_code: Optional[str]
    insights: Optional[str]
    report: Optional[str]

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Intent Agent
def intent_node(state: AgentState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI Analyst dispatcher. Classify the user query into one of these intents:\n"
                   "- 'sql': Needs querying the database (e.g., questions about sales, invoices, customers).\n"
                   "- 'conversational': General conversation, greeting, or clarifying question.\n"
                   "- 'report': Asking to build or export a full summary/report.\n"
                   "Respond with a single JSON object containing 'intent' field."),
        ("user", "{query}")
    ])
    response = llm.invoke(prompt.format(query=state["query"]))
    try:
        data = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        return {"intent": data.get("intent", "sql")}
    except Exception:
        return {"intent": "sql"}

# SQL Generator & Validator Node
def sql_node(state: AgentState) -> Dict[str, Any]:
    if state["intent"] != "sql" and state["intent"] != "report":
        return {}

    schema_info = """
    Table 'customers':
    - customer_id (INTEGER, PK)
    - customer_name (VARCHAR)
    - country (VARCHAR)
    - credit_limit (FLOAT)
    - risk_score (FLOAT)

    Table 'invoices':
    - invoice_id (INTEGER, PK)
    - customer_id (INTEGER, FK to customers)
    - invoice_date (DATE)
    - due_date (DATE)
    - amount (FLOAT)
    - status (VARCHAR)

    Table 'payments':
    - payment_id (INTEGER, PK)
    - invoice_id (INTEGER, FK to invoices)
    - payment_date (DATE)
    - payment_amount (FLOAT)
    """

    # Generate SQL
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a PostgreSQL expert. Based on the database schema:\n{schema_info}\n"
                   "Generate a valid PostgreSQL SELECT query to answer the user query.\n"
                   "Ensure the query is safe (READ-ONLY) and uses correct joins.\n"
                   "Respond with a JSON containing the key 'sql' (just the SQL string)."),
        ("user", "{query}")
    ])
    gen_response = llm.invoke(gen_prompt.format(query=state["query"]))
    try:
        data = json.loads(gen_response.content.strip().replace("```json", "").replace("```", ""))
        sql = data.get("sql", "")
    except Exception:
        # Fallback regex extraction
        sql = gen_response.content

    # SQL Validator (Security check)
    is_safe = True
    sql_lower = sql.lower()
    dangerous_keywords = ["drop", "delete", "insert", "update", "alter", "truncate", "grant", "revoke"]
    for keyword in dangerous_keywords:
        if f" {keyword} " in f" {sql_lower} ":
            is_safe = False
            return {"sql_query": sql, "sql_valid": False, "sql_error": f"Security Alert: Dangerous keyword '{keyword}' detected."}

    return {"sql_query": sql, "sql_valid": is_safe, "sql_error": None}

# Execute Node
def execute_node(state: AgentState) -> Dict[str, Any]:
    if not state.get("sql_valid") or not state.get("sql_query"):
        return {}

    try:
        with engine.connect() as connection:
            result = connection.execute(text(state["sql_query"]))
            cols = result.keys()
            rows = [dict(zip(cols, row)) for row in result.fetchall()]
            return {"query_results": rows, "sql_error": None}
    except Exception as e:
        return {"query_results": None, "sql_error": str(e)}

# Visualization Agent
def visualization_node(state: AgentState) -> Dict[str, Any]:
    if not state.get("query_results"):
        return {}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a charting expert. Based on the query results:\n{results}\n"
                   "Suggest an ECharts configuration for visualizing this data.\n"
                   "Respond with a JSON containing keys 'chart_type' (e.g. bar, line, pie, scatter) and 'option' (valid JSON ECharts option)."),
        ("user", "{query}")
    ])
    results_str = json.dumps(state["query_results"][:10], default=str)
    response = llm.invoke(prompt.format(results=results_str, query=state["query"]))
    try:
        # Just pass the JSON string recommendation
        return {"visualization_code": response.content.strip()}
    except Exception:
        return {}

# Insights Node
def insights_node(state: AgentState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Business AI Analyst. Analyze the data results and provide executive business recommendations:\n"
                   "Results: {results}\n"
                   "SQL Query: {sql}"),
        ("user", "{query}")
    ])
    results_str = json.dumps(state.get("query_results", []), default=str)
    response = llm.invoke(prompt.format(results=results_str, sql=state.get("sql_query", "N/A"), query=state["query"]))
    return {"insights": response.content.strip()}

# Build the LangGraph workflow
def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("intent", intent_node)
    builder.add_node("sql", sql_node)
    builder.add_node("execute", execute_node)
    builder.add_node("visualize", visualization_node)
    builder.add_node("insights", insights_node)

    builder.set_entry_point("intent")
    
    # Define execution routing
    def route_intent(state: AgentState):
        if state["intent"] in ["sql", "report"]:
            return "sql"
        return END

    def route_sql(state: AgentState):
        if state.get("sql_valid"):
            return "execute"
        return END

    builder.add_conditional_edges("intent", route_intent, {"sql": "sql", END: END})
    builder.add_conditional_edges("sql", route_sql, {"execute": "execute", END: END})
    builder.add_edge("execute", "visualize")
    builder.add_edge("visualize", "insights")
    builder.add_edge("insights", END)

    return builder.compile()

# Instantiated workflow
workflow = build_graph()
