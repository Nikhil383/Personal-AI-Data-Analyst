import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_analysis.analyst import load_data, suggest_prompts, prompt_to_code, run_code, ask_llm, get_df_context
import pandas as pd

st.set_page_config(page_title="Personal AI Data Analyst", layout="wide")
st.title("Personal AI Data Analyst — Interactive Dashboard")

st.sidebar.header("Settings")
use_llm = st.sidebar.checkbox("Use LLM for custom prompts", value=False)
provider = "ollama"
api_key = None

if use_llm:
    provider_options = ["Local (Ollama)", "Google Gemini"]
    selected_provider = st.sidebar.radio("Model Provider", provider_options)

    if "Gemini" in selected_provider:
        provider = "gemini"
        llm_model = st.sidebar.text_input("Gemini Model", value="gemini-pro")
        api_key = st.sidebar.text_input("Google API Key", type="password")
        if not api_key:
            st.sidebar.warning("Enter API Key to use Gemini.")
    else:
        provider = "ollama"
        llm_model = st.sidebar.text_input("Ollama Model", value="llama3.1")
        st.sidebar.markdown("Ensure Ollama is installed and running.")

uploaded = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv","xls","xlsx","json"])
if uploaded is None:
    st.info("Upload a CSV / XLSX / JSON to get started. Suggestions will appear automatically.")
    st.stop()

# Load data
try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.success("File loaded.")
with st.expander("Preview data (first 100 rows)"):
    st.dataframe(df.head(100))

# Generate suggestions
suggestions = suggest_prompts(df)
st.markdown("## Suggested analyses (pick one or write your own)")
col1, col2 = st.columns([3,1])
with col1:
    selected = st.selectbox("Choose a suggested prompt", options=suggestions)
    custom = st.text_area("Or write a custom prompt (leave blank to use the selected suggestion)", height=80)
with col2:
    st.markdown("**Quick actions**")
    if st.button("Show suggestions again"):
        st.write(suggestions)

# Determine final prompt
final_prompt = custom.strip() if custom and custom.strip() else selected

st.markdown("### Final prompt")
st.write(final_prompt)

# Run button
if st.button("Run analysis"):
    with st.spinner("Running..."):
        # First try deterministic conversion
        code = prompt_to_code(final_prompt, df)
        if code:
            res = run_code(df, code)
        else:
            # No deterministic code found. If user requested LLM, send the prompt.
            if use_llm:
                # craft a system instruction that asks for python in a ```python``` block that uses df, pd, plt
                context = get_df_context(df)
                system = (
                    "You are a helpful data analyst and will respond with Python code only.\n"
                    "You must return code inside a ```python ... ``` block. The DataFrame is named `df`.\n"
                    "Use pandas for data manipulation and matplotlib for charts. Do not import heavy libs.\n"
                    "If returning a chart, produce matplotlib code that draws the figure (no show()) and nothing else.\n"
                    "Here is the dataset schema:\n"
                    f"{context}"
                )
                raw = system + "\n# User prompt: " + final_prompt
                llm_out = ask_llm(raw, model=llm_model, provider=provider, api_key=api_key)
                if llm_out.startswith("[LLM-missing]") or llm_out.startswith("[LLM-"):
                    st.warning("LLM unavailable or returned an error. Falling back to built-in behavior is not possible for this custom prompt.")
                    st.write(llm_out)
                    st.stop()
                # attempt to extract python block
                if "```python" in llm_out:
                    try:
                        code = llm_out.split("```python")[1].split("```")[0]
                        res = run_code(df, code)
                    except Exception as e:
                        st.error(f"Failed to execute code from LLM: {e}")
                        st.write(llm_out)
                        st.stop()
                else:
                    st.error("LLM did not return a python code block. Showing raw LLM output:")
                    st.write(llm_out)
                    st.stop()
            else:
                st.error("This is a custom prompt that the app cannot deterministically convert to code. Enable 'Use local LLM' in the sidebar to let a local model generate Python, or edit your prompt to match one of the suggested patterns.")
                st.stop()

    # Display result
    if res["type"] == "text":
        st.markdown("#### Output (text)")
        st.text(res["output"])
    elif res["type"] == "dataframe":
        st.markdown("#### Output (table)")
        st.dataframe(res["df"])
        # Provide CSV download
        csv = res["df"].to_csv(index=False).encode("utf-8")
        st.download_button("Download result as CSV", data=csv, file_name="result.csv", mime="text/csv")
    elif res["type"] == "image":
        st.markdown("#### Output (chart)")
        st.image(res["path"], use_column_width=True)
    else:
        st.write("Unknown result type", res)