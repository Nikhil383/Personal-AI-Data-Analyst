import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(
    page_title="Enterprise AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme styling
st.markdown("""
<style>
    .reportview-container {
        background: #0b0f19;
    }
    .sidebar .sidebar-content {
        background: #0f172a;
    }
</style>
""", unsafe_style_allowed=True)

st.title("📊 Enterprise AI Data Analyst (Streamlit Frontend)")
st.write("Query database schemas using natural language, validate safe SQL, and view executive business insights.")

BACKEND_URL = "http://localhost:8000"

# Fetch schema from Backend
schema = {"tables": {}}
try:
    schema_res = requests.get(f"{BACKEND_URL}/api/schema")
    if schema_res.status_code == 200:
        schema = schema_res.json()
except Exception:
    st.sidebar.warning("Could not connect to FastAPI backend at http://localhost:8000")

# Sidebar
st.sidebar.header("Database Schema")

if st.sidebar.button("Seed Mock Database", type="primary"):
    with st.spinner("Initializing and seeding database..."):
        try:
            init_res = requests.post(f"{BACKEND_URL}/api/db/init")
            if init_res.status_code == 200:
                st.sidebar.success(init_res.json().get("message", "DB Seeded successfully!"))
                # Refresh schema
                schema_res = requests.get(f"{BACKEND_URL}/api/schema")
                if schema_res.status_code == 200:
                    schema = schema_res.json()
            else:
                st.sidebar.error("Failed to seed database.")
        except Exception as e:
            st.sidebar.error(f"Error connecting: {e}")

# Display tables in sidebar
for table, cols in schema.get("tables", {}).items():
    with st.sidebar.expander(f"📁 {table}"):
        for col in cols:
            st.write(f"🔹 {col}")

# Chat/Query interface
query = st.text_input("Ask a question about your database:", placeholder="e.g. Compare total amount of invoices by status")

if query:
    with st.spinner("Executing analytical agents..."):
        try:
            res = requests.post(f"{BACKEND_URL}/api/query", json={"query": query})
            if res.status_code == 200:
                data = res.json()
                
                # Split page layout
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Query Execution & Data")
                    
                    # SQL & Validation
                    if data.get("sql_query"):
                        st.markdown("### Generated SQL")
                        if data.get("sql_valid"):
                            st.success("✓ Safe Read-Only SQL Query")
                        else:
                            st.error("⚠️ Security Warning: Unsafe queries blocked")
                        
                        st.code(data["sql_query"], language="sql")
                        
                    # Query Results
                    if data.get("query_results"):
                        st.markdown("### Query Results")
                        results_df = pd.DataFrame(data["query_results"])
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Fallback visualization using Streamlit charts if data has columns
                        if len(results_df.columns) >= 2:
                            st.markdown("### Visualization")
                            numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
                            cat_cols = results_df.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            if cat_cols and numeric_cols:
                                st.bar_chart(results_df.set_index(cat_cols[0])[numeric_cols[0]])
                            elif len(numeric_cols) >= 2:
                                st.line_chart(results_df[numeric_cols[:2]])
                            elif numeric_cols:
                                st.bar_chart(results_df[numeric_cols[0]])
                
                with col2:
                    st.subheader("AI Business Insights")
                    if data.get("insights"):
                        st.info(data["insights"])
                    else:
                        st.write("No insights generated.")
                        
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Error executing agent pipeline: {e}")
