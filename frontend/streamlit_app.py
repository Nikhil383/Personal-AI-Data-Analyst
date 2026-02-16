import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import dotenv
import os
import sys

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_analysis.graph import create_agentic_workflow
from src.data_analysis.utils import APIKeyValidator, is_google_api_key_present, get_google_api_key
from src.utils.plot_utils import get_plot_as_image

# Load env variables (API key)
dotenv.load_dotenv()

st.set_page_config(page_title="🤖 Agentic AI Data Analyst", page_icon="📊", layout="wide")

# Custom CSS for agent badges
st.markdown("""
<style>
.agent-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.875rem;
    font-weight: 600;
    margin: 0.25rem;
}
.supervisor { background-color: #667eea; color: white; }
.data-cleaner { background-color: #48bb78; color: white; }
.stats-analyst { background-color: #ed8936; color: white; }
.visualizer { background-color: #9f7aea; color: white; }
.query-answerer { background-color: #4299e1; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic AI Data Analyst")
st.caption("Powered by LangGraph, LangChain & Google Gemini")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Setup & Tools
with st.sidebar:
    st.header("⚙️ Data Setup")
    uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx", "xls"])
    
    # Use API key validator
    api_key_valid = is_google_api_key_present()
    api_key = get_google_api_key()
    
    if api_key_valid:
        st.success("✅ Google API Key Loaded")
    else:
        st.error("❌ Google API Key Missing")
        st.warning("Set GOOGLE_API_KEY in your .env file")
    
    # Show all API key statuses
    with st.expander("🔑 API Key Status"):
        validation_results = APIKeyValidator.validate_all_keys()
        for key_type, info in validation_results.items():
            status = "✅" if info['valid'] else "❌"
            st.write(f"{status} **{info['name']}**")
    
    st.divider()
    
    if uploaded_file:
        try:
            # Load Data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {uploaded_file.name}")
            st.info(f"📊 {len(df)} rows × {len(df.columns)} columns")
            st.session_state['df'] = df
            
            # Initialize Agentic Workflow
            if api_key:
                try:
                    workflow = create_agentic_workflow(df, api_key=api_key)
                    st.session_state['workflow'] = workflow
                    st.success("🤖 Multi-Agent System Ready!")
                except Exception as e:
                    st.error(f"Workflow Init Failed: {e}")
            else:
                st.warning("Please set GOOGLE_API_KEY in .env file")
                
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Dataset Tools
    if 'df' in st.session_state:
        st.divider()
        st.subheader("📋 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Summary", use_container_width=True):
                st.write(st.session_state['df'].describe())
        
        with col2:
            if st.button("🔍 Info", use_container_width=True):
                st.write(f"**Shape:** {st.session_state['df'].shape}")
                st.write(f"**Columns:** {', '.join(st.session_state['df'].columns)}")
        
        st.divider()
        st.subheader("🤖 Active Agents")
        st.markdown("""
        <div class="agent-badge supervisor">🎯 Supervisor</div>
        <div class="agent-badge data-cleaner">🧹 Data Cleaner</div>
        <div class="agent-badge stats-analyst">📊 Stats Analyst</div>
        <div class="agent-badge visualizer">📈 Visualizer</div>
        <div class="agent-badge query-answerer">💬 Query Answerer</div>
        """, unsafe_allow_html=True)

# Main chat interface
st.subheader("💬 Chat with Your Data")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show agent badge if present
        if "agent" in msg:
            agent_name = msg["agent"].replace("_", " ").title()
            agent_class = msg["agent"].replace("_", "-")
            st.markdown(f'<div class="agent-badge {agent_class}">{agent_name}</div>', unsafe_allow_html=True)
        
        st.write(msg["content"])
        
        # Show reasoning if present
        if "reasoning" in msg:
            with st.expander("🧠 Agent Reasoning"):
                st.write(msg["reasoning"])
        
        # Show image if present
        if "image" in msg:
            st.image(msg["image"])

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Agent Response
    if 'workflow' in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agents working..."):
                try:
                    # Clear any existing plots
                    plt.clf()
                    plt.close('all')
                    
                    # Invoke the agentic workflow
                    response = st.session_state['workflow'].invoke(prompt)
                    
                    # Extract response
                    output_text = response.get("output", "No response generated")
                    agent_name = response.get("agent", "unknown")
                    
                    # Show agent badge
                    agent_display = agent_name.replace("_", " ").title()
                    agent_class = agent_name.replace("_", "-")
                    st.markdown(f'<div class="agent-badge {agent_class}">🤖 {agent_display}</div>', unsafe_allow_html=True)
                    
                    # Show output
                    st.write(output_text)
                    
                    # Show reasoning steps if available
                    if "messages" in response:
                        reasoning_messages = [
                            msg.content for msg in response["messages"] 
                            if hasattr(msg, 'content') and msg.content.startswith("[")
                        ]
                        if reasoning_messages:
                            with st.expander("🧠 Agent Reasoning Steps"):
                                for reasoning in reasoning_messages:
                                    st.write(reasoning)
                    
                    # Check for plot
                    fig = plt.gcf()
                    img_buf = None
                    if fig.get_axes():
                        img_buf = get_plot_as_image(fig)
                        st.image(img_buf)
                    
                    # Save to history
                    msg_data = {
                        "role": "assistant", 
                        "content": output_text,
                        "agent": agent_name
                    }
                    if img_buf:
                        msg_data["image"] = img_buf
                    if reasoning_messages:
                        msg_data["reasoning"] = "\n".join(reasoning_messages)
                    
                    st.session_state.messages.append(msg_data)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.exception(e)
    else:
        st.warning("⚠️ Please upload a file and ensure API Key is set to start analyzing.")

# Footer
st.divider()
st.caption("💡 **Tip:** Try asking to clean data, analyze correlations, create visualizations, or ask general questions!")
