import streamlit as st
import requests
import pandas as pd
import json

# Setup page
st.set_page_config(page_title="Personal AI Data Analyst", page_icon="📊", layout="wide")

API_URL = "http://localhost:8000"

# Styling
st.markdown("""
<style>
    .stChatInput { position: fixed; bottom: 3rem; }
    .main { padding-bottom: 5rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Data Analyst V2")
st.caption("Powered by Gemini 1.5 & Pandas")

# Sidebar - Upload
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "json"])
    
    if uploaded_file and "file_uploaded" not in st.session_state:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        with st.spinner("Uploading & Analyzing..."):
            try:
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Loaded {data['filename']}")
                    st.info(f"**Initial Insight:**\n\n{data['initial_insight']}")
                    st.session_state["file_uploaded"] = True
                    st.session_state["columns"] = data['columns']
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    if "columns" in st.session_state:
        st.write("### Columns")
        st.write(st.session_state["columns"])

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Upload a dataset and ask me anything."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], dict): 
            # Handle structured responses (images, dataframes)
            content = msg["content"]
            if content.get("text"):
                st.markdown(content["text"])
            if content.get("image"):
                st.image(content["image"])
                if content.get("vision_analysis"):
                    st.info(f"👁️ **Vision Analysis:** {content['vision_analysis']}")
            if content.get("dataframe"):
                st.dataframe(pd.DataFrame(content["dataframe"]))
        else:
            st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Analyze trends, plot graphs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                res = requests.post(f"{API_URL}/analyze", json={"prompt": prompt})
                response = res.json()
                
                # Format output for display & history
                history_content = {}
                
                if response.get("type") == "error":
                    st.error(response["error"])
                    history_content["text"] = f"❌ Error: {response['error']}"
                    
                elif response.get("type") == "text":
                    st.markdown(response["output"])
                    history_content["text"] = response["output"]
                    
                elif response.get("type") == "dataframe":
                    df_data = response["df"]
                    st.dataframe(df_data)
                    st.markdown(f"*{response.get('logs', '')}*")
                    history_content["dataframe"] = df_data
                    
                elif response.get("type") == "image":
                    st.image(response["path"])
                    if response.get("vision_analysis"):
                        st.info(response["vision_analysis"])
                    history_content["image"] = response["path"]
                    history_content["vision_analysis"] = response.get("vision_analysis")
                
                st.session_state.messages.append({"role": "assistant", "content": history_content})
                
            except Exception as e:
                st.error(f"Failed to communicate with analyst backend: {e}")
