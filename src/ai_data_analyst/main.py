"""AI Data Analyst - Main Application"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path

from config import GOOGLE_API_KEY, CHARTS_DIR, DATA_DIR
from data_loader import DataLoader
from analyzer import DataAnalyzer
from visualizer import DataVisualizer
from chains import AnalystChain


# Page configuration
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0D1117;
    }

    /* Headers */
    h1, h2, h3 {
        color: #E6EDF3 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }

    /* Cards */
    .css-1r6slb0, .stCard {
        background-color: #161B22;
        border: 1px solid rgba(0, 212, 170, 0.1);
        border-radius: 8px;
        padding: 16px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D4AA 0%, #2D5A87 100%);
        border: none;
        border-radius: 6px;
        color: #0D1117;
        font-weight: bold;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(0, 212, 170, 0.5);
        transform: translateY(-1px);
    }

    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #161B22;
        border: 1px solid #2D5A87;
        color: #E6EDF3;
        border-radius: 6px;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00D4AA;
        box-shadow: 0 0 5px rgba(0, 212, 170, 0.3);
    }

    /* File uploader */
    .stFileUploader {
        background-color: #161B22;
        border: 2px dashed #2D5A87;
        border-radius: 8px;
        padding: 20px;
    }

    /* DataFrame */
    .dataframe {
        background-color: #161B22 !important;
        color: #E6EDF3 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
    }
    .dataframe th {
        background-color: #1E3A5F !important;
        color: #00D4AA !important;
    }
    .dataframe td {
        border-color: #2D5A87 !important;
    }
    .dataframe tr:nth-child(even) {
        background-color: #0D1117 !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #161B22;
    }

    /* Messages */
    .user-message {
        background-color: #1E3A5F;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid #00D4AA;
    }
    .assistant-message {
        background-color: #161B22;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid #2D5A87;
    }

    /* Success/Error/Info messages */
    .stSuccess {
        background-color: rgba(63, 185, 80, 0.1);
        border: 1px solid #3FB950;
    }
    .stError {
        background-color: rgba(248, 81, 73, 0.1);
        border: 1px solid #F85149;
    }
    .stInfo {
        background-color: rgba(0, 212, 170, 0.1);
        border: 1px solid #00D4AA;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 4px 4px 0px 0px;
        color: #8B949E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: #00D4AA !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00D4AA !important;
        font-family: 'JetBrains Mono', monospace;
    }
    [data-testid="stMetricLabel"] {
        color: #8B949E !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'file_loaded' not in st.session_state:
        st.session_state.file_loaded = False


def load_data(uploaded_file):
    """Load data from uploaded file."""
    try:
        # Save uploaded file temporarily
        temp_path = DATA_DIR / uploaded_file.name
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Load data
        df = DataLoader.load_file(str(temp_path))

        # Initialize analyzer and visualizer
        analyzer = DataAnalyzer(df)
        visualizer = DataVisualizer(df, CHARTS_DIR)
        chain = AnalystChain(df, GOOGLE_API_KEY)

        st.session_state.dataframe = df
        st.session_state.analyzer = analyzer
        st.session_state.visualizer = visualizer
        st.session_state.chain = chain
        st.session_state.file_loaded = True

        # Clean up temp file
        os.remove(temp_path)

        return True, "Data loaded successfully!"
    except Exception as e:
        return False, f"Error loading data: {str(e)}"


def display_data_info():
    """Display data information."""
    df = st.session_state.dataframe

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns)}")
    with col3:
        st.metric("Numeric Columns", f"{len(df.select_dtypes(include=['number']).columns)}")
    with col4:
        st.metric("Categorical Columns", f"{len(df.select_dtypes(include=['object', 'category']).columns)}")

    # Data types
    with st.expander("📋 Column Data Types", expanded=False):
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(dtype) for dtype in df.dtypes],
            'Non-Null': [df[col].notna().sum() for col in df.columns],
            'Null': [df[col].isna().sum() for col in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    # Summary statistics
    with st.expander("📊 Summary Statistics", expanded=False):
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns to display statistics for.")


def display_data_preview():
    """Display data preview table."""
    df = st.session_state.dataframe
    st.dataframe(df.head(100), use_container_width=True)


def handle_query(query: str, show_reasoning: bool = False):
    """
    Handle user query using ReAct pattern.

    Workflow: User Question → LangChain Agent → Gemini LLM → Pandas Tool → Answer
    """
    if not query.strip():
        return

    # Add to history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Get response from chain using the new analyze method
    analysis_response = st.session_state.chain.analyze(query)

    # Format the response
    if show_reasoning:
        # Include reasoning steps
        full_response = st.session_state.chain.get_workflow_summary(analysis_response)
        response_content = full_response
    else:
        # Just the answer
        response_content = analysis_response.final_answer

    # Add response to history with metadata
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_content,
        "chart_type": analysis_response.chart_type,
        "chart_columns": analysis_response.chart_columns
    })

    return analysis_response


def display_chat_history():
    """Display chat history."""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">📊 {msg["content"]}</div>', unsafe_allow_html=True)
            # Show chart suggestion if available
            if "chart_type" in msg and msg["chart_type"]:
                with st.expander("📈 Suggested Visualization"):
                    st.write(f"**Chart Type:** {msg['chart_type']}")
                    if "chart_columns" in msg and msg["chart_columns"]:
                        st.write(f"**Columns:** {', '.join(msg['chart_columns'])}")


def main():
    """Main application."""
    initialize_session_state()

    # Header
    st.title("🤖 AI Data Analyst")
    st.markdown("### Your intelligent data analysis assistant")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Source")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel"
        )

        if uploaded_file:
            if not st.session_state.file_loaded:
                with st.spinner("Loading data..."):
                    success, message = load_data(uploaded_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)

            st.divider()

            # Data operations
            st.header("📊 Data Operations")

            # Column selector for analysis
            if st.session_state.analyzer:
                numeric_cols = st.session_state.dataframe.select_dtypes(include=['number']).columns.tolist()
                cat_cols = st.session_state.dataframe.select_dtypes(include=['object', 'category']).columns.tolist()

                if numeric_cols or cat_cols:
                    all_cols = numeric_cols + cat_cols
                    selected_col = st.selectbox("Analyze Column", all_cols)

                    if st.button("🔍 Analyze Column"):
                        with st.spinner("Analyzing..."):
                            analysis = st.session_state.analyzer.get_column_analysis(selected_col)
                            st.json(analysis)

            # Visualization suggestions
            if st.session_state.visualizer:
                st.divider()
                st.header("📈 Visualizations")

                if st.button("💡 Get Suggestions"):
                    suggestions = st.session_state.analyzer.suggest_visualizations()
                    for i, sug in enumerate(suggestions, 1):
                        st.markdown(f"**{i}. {sug['type'].title()}**: {sug['description']}")

        # Clear data button
        if st.session_state.file_loaded:
            st.divider()
            if st.button("🗑️ Clear Data", type="primary"):
                st.session_state.dataframe = None
                st.session_state.analyzer = None
                st.session_state.visualizer = None
                st.session_state.chain = None
                st.session_state.file_loaded = False
                st.session_state.chat_history = []
                st.rerun()

    # Main content
    if not st.session_state.file_loaded:
        # Welcome screen
        st.markdown("""
        ### Welcome to AI Data Analyst! 🚀

        Get started by uploading a data file in the sidebar.

        **Supported file types:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)

        **Features:**
        - 📊 Load and preview data
        - 🔍 Ask questions in natural language
        - 📈 Generate visualizations
        - 💡 Get AI-powered insights

        ---
        """)
    else:
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "💬 Analysis", "📈 Visualizations"])

        with tab1:
            st.subheader("Data Preview")
            display_data_info()
            st.markdown("### First 100 Rows")
            display_data_preview()

        with tab2:
            st.subheader("Ask Questions About Your Data")

            # ReAct workflow explanation
            with st.expander("ℹ️ How it works (ReAct Pattern)", expanded=False):
                st.markdown("""
                **Analysis Workflow:**
                ```
                User Question → LangChain Agent → Gemini LLM (Reasoning) → Pandas Tool → Answer
                ```
                1. **Understand** - Analyze your question
                2. **Reason** - Plan data operations needed
                3. **Act** - Execute pandas operations
                4. **Observe** - Review results
                5. **Answer** - Provide natural language response
                """)

            # Show reasoning toggle
            show_reasoning = st.checkbox("🔍 Show reasoning steps", value=False,
                                        help="Display the step-by-step reasoning process")

            # Display chat history
            display_chat_history()

            # Query input
            query = st.text_area(
                "What would you like to know about your data?",
                placeholder="e.g., What is the average sales value? or Show me the top 5 products by revenue",
                height=100,
                key="query_input"
            )

            col1, col2 = st.columns([1, 5])
            with col1:
                send_btn = st.button("🔍 Analyze", type="primary")
            with col2:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()

            if send_btn and query:
                with st.spinner("🤔 Thinking... Analyzing your data..."):
                    response = handle_query(query, show_reasoning=show_reasoning)
                    st.rerun()

        with tab3:
            st.subheader("Create Visualizations")

            if st.session_state.visualizer:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["histogram", "bar", "scatter", "line", "box", "pie", "correlation"]
                )

                df = st.session_state.dataframe
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

                if chart_type == "histogram":
                    col = st.selectbox("Select Column", numeric_cols)
                    if st.button("📊 Create Histogram"):
                        with st.spinner("Creating..."):
                            img_path = st.session_state.visualizer.create_histogram(col)
                            st.image(img_path, use_container_width=True)

                elif chart_type == "bar":
                    x_col = st.selectbox("Select Category Column", cat_cols)
                    y_col = st.selectbox("Select Value Column (optional)", ["None"] + numeric_cols)
                    y_col = None if y_col == "None" else y_col

                    if st.button("📊 Create Bar Chart"):
                        with st.spinner("Creating..."):
                            img_path = st.session_state.visualizer.create_bar_chart(x_col, y_col)
                            st.image(img_path, use_container_width=True)

                elif chart_type == "scatter":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X Axis", numeric_cols, key="scatter_x")
                    with col2:
                        y_col = st.selectbox("Y Axis", numeric_cols, key="scatter_y")

                    if st.button("📊 Create Scatter Plot"):
                        with st.spinner("Creating..."):
                            img_path = st.session_state.visualizer.create_scatter(x_col, y_col)
                            st.image(img_path, use_container_width=True)

                elif chart_type == "line":
                    x_col = st.selectbox("X Axis (Category/Time)", cat_cols + numeric_cols[:1])
                    y_cols = st.multiselect("Y Axis (Values)", numeric_cols, default=numeric_cols[:1] if numeric_cols else [])

                    if st.button("📊 Create Line Chart") and y_cols:
                        with st.spinner("Creating..."):
                            img_path = st.session_state.visualizer.create_line_chart(x_col, y_cols)
                            st.image(img_path, use_container_width=True)

                elif chart_type == "box":
                    cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)

                    if st.button("📊 Create Box Plot") and cols:
                        with st.spinner("Creating..."):
                            img_path = st.session_state.visualizer.create_box_plot(cols)
                            st.image(img_path, use_container_width=True)

                elif chart_type == "pie":
                    col = st.selectbox("Select Column", cat_cols)
                    top_n = st.slider("Top N Categories", 3, 10, 5)

                    if st.button("📊 Create Pie Chart"):
                        with st.spinner("Creating..."):
                            img_path = st.session_state.visualizer.create_pie_chart(col, top_n)
                            st.image(img_path, use_container_width=True)

                elif chart_type == "correlation":
                    if st.button("📊 Create Correlation Heatmap"):
                        with st.spinner("Creating..."):
                            try:
                                img_path = st.session_state.visualizer.create_correlation_heatmap()
                                st.image(img_path, use_container_width=True)
                            except ValueError as e:
                                st.error(str(e))

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #8B949E;'>"
        "AI Data Analyst powered by LangChain & Gemini API"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
