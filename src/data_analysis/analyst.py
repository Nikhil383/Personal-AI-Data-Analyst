import io
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys
import os
import contextlib
import traceback
import uuid
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


# Optional duckdb import (not required)
try:
    import duckdb
except Exception:
    duckdb = None

# ----------------- Load data -----------------
def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(errors="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample


def load_data(file_or_path) -> pd.DataFrame:
    """
    Accepts Streamlit UploadedFile, path string/Path, or file-like object.
    Returns pandas DataFrame.
    """
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        s = p.suffix.lower()
        if s == ".csv":
            return pd.read_csv(p)
        if s in {".xls", ".xlsx"}:
            return pd.read_excel(p)
        if s == ".json":
            return pd.read_json(p)
        return pd.read_csv(p)

    # file-like (UploadedFile)
    name = getattr(file_or_path, "name", None)
    suffix = Path(name).suffix.lower() if name else None
    raw = file_or_path.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    bio = io.BytesIO(raw)

    if suffix == ".csv" or (suffix is None and _looks_like_csv(raw)):
        bio.seek(0); return pd.read_csv(bio)
    if suffix in {".xls", ".xlsx"}:
        bio.seek(0); return pd.read_excel(bio)
    if suffix == ".json":
        bio.seek(0); return pd.read_json(bio)
    # fallback
    bio.seek(0)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0); return pd.read_json(bio)


def get_df_context(df: pd.DataFrame) -> str:
    """
    Returns a string summary of the DataFrame columns and types for the LLM.
    """
    buffer = io.StringIO()
    buffer.write("DataFrame Columns:\n")
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "int" in dtype or "float" in dtype:
             # Add range for numeric to help LLM know boundaries
             min_val = df[col].min()
             max_val = df[col].max()
             buffer.write(f"- {col} ({dtype}): min={min_val}, max={max_val}\n")
        elif "object" in dtype or "category" in dtype:
             # Add a few unique values for categoricals
             unique = df[col].dropna().unique()
             if len(unique) < 10:
                 ex = ", ".join(map(str, unique))
             else:
                 ex = ", ".join(map(str, unique[:5])) + "..."
             buffer.write(f"- {col} ({dtype}): examples=[{ex}]\n")
        else:
             buffer.write(f"- {col} ({dtype})\n")
    return buffer.getvalue()


def _detect_column_types(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime = []
    # try to infer datetime columns
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            datetime.append(c)
        else:
            # try to parse small sample as date
            try:
                sample = df[c].dropna().astype(str).iloc[:20]
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= max(1, min(5, len(sample)//2)):
                    datetime.append(c)
            except Exception:
                pass
    # categoricals: low cardinality non-numeric
    categorical = [c for c in df.columns if c not in numeric + datetime and df[c].nunique(dropna=True) <= 50]
    return {"numeric": numeric, "datetime": datetime, "categorical": categorical}


def suggest_prompts(df: pd.DataFrame, max_suggestions: int = 8):
    """
    Return a list of helpful, ready-to-run prompt strings for the dataset.
    Deterministic and works without any LLM.
    """
    types = _detect_column_types(df)
    numeric = types["numeric"]
    datetime = types["datetime"]
    categorical = types["categorical"]

    suggestions = []
    # Basic summary
    suggestions.append("Summarize the dataset in 5 bullet points (rows, columns, missing values, numeric columns, top categorical).")
    # Top value queries
    if categorical:
        col = categorical[0]
        suggestions.append(f"Show the top 10 counts for the categorical column '{col}'.")
    # Numeric summaries
    if numeric:
        suggestions.append(f"Show summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.")
        col = numeric[0]
        suggestions.append(f"Create a histogram of the numeric column '{col}'.")
        if len(numeric) >= 2:
            suggestions.append(f"Create a scatter plot comparing '{numeric[0]}' (x) vs '{numeric[1]}' (y).")
        suggestions.append(f"Show the top 10 rows sorted by '{col}' descending.")
    # Time series
    if datetime:
        dcol = datetime[0]
        # choose a numeric for aggregation if exists
        ag = numeric[0] if numeric else None
        if ag:
            suggestions.append(f"Create a time series of monthly sum of '{ag}' using the datetime column '{dcol}'.")
        else:
            suggestions.append(f"Show counts per month using the datetime column '{dcol}'.")
    # Correlation
    if len(numeric) >= 2:
        suggestions.append("Show the correlation matrix heatmap for numeric columns.")
    # Generic top-k
    suggestions.append("Find rows that look like anomalies using z-score > 3 on numeric columns and show top 20.")
    # limit suggestions
    return suggestions[:max_suggestions]


def prompt_to_code(prompt: str, df: pd.DataFrame):
    """
    Convert known prompt templates into runnable python code strings.
    If the prompt is custom/unrecognized, return None (so UI can send to LLM instead).
    """
    p = prompt.strip().lower()

    # Summary
    if p.startswith("summarize the dataset"):
        code = textwrap.dedent("""
            # produce a short summary as printed text
            info = []
            info.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            info.append("Column types: " + ", ".join([f"{c}:{str(df[c].dtype)[:10]}" for c in df.columns[:10]]))
            miss = df.isnull().sum().sort_values(ascending=False).head(10)
            info.append("Top missing: " + ", ".join([f"{idx}:{val}" for idx,val in miss.items() if val>0]))
            numeric = df.select_dtypes(include=['number']).columns.tolist()
            info.append(f"Numeric columns count: {len(numeric)}")
            # print concise bullets
            result = "\\n".join(["- "+i for i in info])
        """)
        return code

    # Top counts for categorical
    if "top 10 counts for the categorical column" in p or "top 10 counts" in p and "'" in p:
        # try to extract column name between quotes
        import re
        m = re.search(r"'([^']+)'", prompt)
        if not m:
            m = re.search(r'"([^\"]+)"', prompt)
        col = m.group(1) if m else None
        if col:
            code = textwrap.dedent(f"""
                # top 10 counts for '{col}'
                result = df['{col}'].value_counts(dropna=False).head(10).reset_index()
                result.columns = ['value','count']
            """)
            return code

    # Summary statistics for numeric
    if "summary statistics" in p or "describe" in p:
        code = textwrap.dedent("""
            result = df.select_dtypes(include=['number']).describe().T
        """)
        return code

    # Histogram
    if p.startswith("create a histogram of the numeric column") or "histogram of the numeric column" in p:
        import re
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col:
            code = textwrap.dedent(f"""
                # histogram for '{col}'
                plt.figure(figsize=(6,4))
                df['{col}'].dropna().astype(float).hist(bins=30)
                plt.title('Histogram of {col}')
                plt.xlabel('{col}')
                plt.ylabel('count')
                # produce an image by saving to result_img_path variable
                result_img_path = None
            """)
            # We'll return plotting code that uses plt; execution will save figure
            return code

    # Scatter plot
    if "scatter plot comparing" in p and "vs" in p:
        import re
        m = re.search(r"'([^']+)' \\(x\\) vs '([^']+)' \\(y\\)", prompt)
        if m:
            xcol, ycol = m.group(1), m.group(2)
            code = textwrap.dedent(f"""
                plt.figure(figsize=(6,4))
                df.plot.scatter(x='{xcol}', y='{ycol}')
                plt.title('{ycol} vs {xcol}')
                result_img_path = None
            """)
            return code

    # Top N rows sorted by col
    if p.startswith("show the top 10 rows sorted by"):
        import re
        m = re.search(r"by '([^']+)'", prompt)
        if m:
            col = m.group(1)
            code = textwrap.dedent(f"""
                result = df.sort_values('{col}', ascending=False).head(10).reset_index(drop=True)
            """)
            return code

    # Time series monthly sum
    if "monthly sum" in p and "using the datetime column" in p:
        import re
        m = re.search(r"sum of '([^']+)' using the datetime column '([^']+)'", prompt)
        if m:
            ag, dcol = m.group(1), m.group(2)
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}'])
                res = res.set_index('{dcol}').resample('M')['{ag}'].sum().reset_index()
                result = res
            """)
            return code

    # Counts per month (datetime only)
    if "counts per month using the datetime column" in p:
        import re
        m = re.search(r"datetime column '([^']+)'", prompt)
        dcol = m.group(1) if m else None
        if dcol:
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}']).set_index('{dcol}').resample('M').size().reset_index(name='count')
                result = res
            """)
            return code

    # Correlation heatmap
    if "correlation matrix heatmap" in p or "correlation heatmap" in p:
        code = textwrap.dedent("""
            corr = df.select_dtypes(include=['number']).corr()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,5))
            plt.imshow(corr, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Correlation matrix')
            result_img_path = None
        """)
        return code

    # Anomaly detection using z-score
    if "anomalies" in p and "z-score" in p:
        code = textwrap.dedent("""
            from scipy import stats
            num = df.select_dtypes(include=['number']).dropna()
            if num.shape[1]==0:
                result = pd.DataFrame()
            else:
                z = np.abs(stats.zscore(num.select_dtypes(include=['number'])))
                mask = (z > 3).any(axis=1)
                result = df.loc[mask].head(20).reset_index(drop=True)
        """)
        return code

    # Unknown / custom prompts -> return None
    return None


def ask_llm(prompt: str, model: str = "llama3.1", provider: str = "ollama", api_key: str = None) -> str:
    """
    Dispatch prompt to the selected LLM provider.
    """
    if provider == "gemini":
        return ask_gemini(prompt, api_key, model)
    else:
        # Default to ollama
        return ask_ollama(prompt, model)


def ask_ollama(prompt: str, model: str = "llama3.1", timeout: int = 60) -> str:
    """
    Send prompt to local Ollama via CLI. Returns stdout text.
    If ollama is not installed or fails, returns an error string starting with [LLM...].
    Expect the model to return code inside ```python blocks.
    """
    try:
        proc = subprocess.run(["ollama", "run", model], input=prompt.encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        if not out and err:
            return f"[LLM-error] {err}"
        return out
    except FileNotFoundError:
        return "[LLM-missing] ollama not found on PATH."
    except Exception as e:
        return f"[LLM-failed] {e}"


def ask_gemini(prompt: str, api_key: str, model: str = "gemini-pro") -> str:
    """
    Send prompt to Google Gemini via LangChain.
    """
    if not ChatGoogleGenerativeAI:
        return "[LLM-error] langchain-google-genai not installed."
    if not api_key:
        return "[LLM-error] No Google API Key provided."
    
    try:
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, convert_system_message_to_human=True)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"[LLM-failed] {str(e)}"



# ----------------- Execute generated code safely -----------------
def run_code(df: pd.DataFrame, code: str, timeout_seconds: int = 30):
    """
    Execute a python code string that can use `df`, `pd`, `plt`, `np`.
    Returns a dict like:
      - {"type":"text", "output": "..." }
      - {"type":"dataframe", "df": pd.DataFrame(...) }
      - {"type":"image", "path": "/tmp/....png" }
    Execution captures printed text and looks for:
      - `result` variable (DataFrame/Series/str)
      - `result_img_path` variable (path to image) OR if matplotlib created a figure it saves that to a temp png file.
    NOTE: This executes arbitrary code. Only run on trusted inputs / locally.
    """
    # prepare namespace for execution
    exec_globals = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df,
    }
    exec_locals = {}

    # capture stdout and exceptions
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            # execute the code (use exec so multi-line works)
            exec(code, exec_globals, exec_locals)
    except Exception as e:
        # include traceback to help debugging in the app
        tb = traceback.format_exc()
        return {"type": "text", "output": f"[Execution error]\n{tb}"}

    printed = buf.getvalue().strip()

    # check for result_img_path set by user code
    result_img_path = exec_locals.get("result_img_path", None) or exec_globals.get("result_img_path", None)

    # If an image path was provided by code, return it
    if result_img_path:
        return {"type": "image", "path": str(result_img_path)}

    # If matplotlib figure(s) were drawn, save the current figure
    try:
        figs = plt.get_fignums()
    except Exception:
        figs = []
    if figs:
        out_file = Path(tempfile.gettempdir()) / f"chart_{uuid.uuid4().hex}.png"
        try:
            plt.savefig(out_file, bbox_inches='tight')
        except Exception as e:
            return {"type": "text", "output": f"[Image save error] {e}\n\nPrinted output:\n{printed}"}
        finally:
            plt.close('all')
        return {"type": "image", "path": str(out_file)}

    # If code produced a `result` variable, use it
    result = exec_locals.get("result", exec_globals.get("result", None))

    # If there's printed output and no result, return printed text
    if result is None:
        if printed:
            return {"type": "text", "output": printed}
        else:
            return {"type": "text", "output": "No output was produced by the executed code."}

    # If result is a DataFrame or Series: return as dataframe
    if isinstance(result, pd.DataFrame):
        return {"type": "dataframe", "df": result}
    if isinstance(result, pd.Series):
        return {"type": "dataframe", "df": result.to_frame().reset_index()}

    # If result is string
    if isinstance(result, str):
        out = result
        if printed:
            out = printed + "\n\n" + out
        return {"type": "text", "output": out}

    # If result is iterable -> try dataframe
    try:
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            try:
                df_res = pd.DataFrame(result)
                return {"type": "dataframe", "df": df_res}
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: just str() the result
    return {"type": "text", "output": (printed + "\n\n" if printed else "") + str(result)}


# ----------------- Small local test runner -----------------
if __name__ == "__main__":
    # quick sanity test you can run with `python analyst.py`
    print("Running quick self-test for analyst.py...\n")
    df_test = pd.DataFrame({
        'A': np.random.randn(200),
        'B': np.random.choice(['x','y','z'], size=200),
        'date': pd.date_range('2025-01-01', periods=200, freq='D')
    })

    print("Suggest prompts:")
    for s in suggest_prompts(df_test):
        print(" -", s)

    prompt = f"Create a histogram of the numeric column 'A'."
    print("\nPrompt->code:", prompt)
    code = prompt_to_code(prompt, df_test)
    if code is None:
        print("prompt_to_code returned None (no deterministic template matched)")
    else:
        print("Generated code (first 300 chars):\n", code[:300])

    print("\nExecuting generated code via run_code()...")
    res = run_code(df_test, code)
    print("Result type:", res.get('type'))
    if res.get('type') == 'image':
        print("Saved image at:", res.get('path'))
    elif res.get('type') == 'dataframe':
        print(res.get('df').head())
    else:
        print(res.get('output'))

    print("\nSelf-test complete.")
