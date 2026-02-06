import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import textwrap
import traceback
import contextlib
from pathlib import Path
import sys
import uuid
import tempfile

class PandasAnalyzer:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def load_data(self, file_or_path) -> pd.DataFrame:
        if isinstance(file_or_path, (str, Path)):
            p = Path(file_or_path)
            if p.suffix.lower() == ".csv": return pd.read_csv(p)
            if p.suffix.lower() in {".xls", ".xlsx"}: return pd.read_excel(p)
            if p.suffix.lower() == ".json": return pd.read_json(p)
            return pd.read_csv(p)
        
        # Streamlit or raw bytes
        if hasattr(file_or_path, "name"):
            name = file_or_path.name
            if name.endswith(".csv"): return pd.read_csv(file_or_path)
            if name.endswith((".xls", ".xlsx")): return pd.read_excel(file_or_path)
            if name.endswith(".json"): return pd.read_json(file_or_path)
        
        # Fallback
        try:
            return pd.read_csv(file_or_path)
        except:
            return pd.read_excel(file_or_path)

    def get_df_context(self, df: pd.DataFrame) -> str:
        buffer = io.StringIO()
        buffer.write("Columns:\n")
        for col in df.columns:
            dtype = str(df[col].dtype)
            if "number" in dtype or "int" in dtype or "float" in dtype:
                min_v = df[col].min()
                max_v = df[col].max()
                buffer.write(f"- {col} ({dtype}): min={min_v}, max={max_v}\n")
            elif "object" in dtype:
                unique = df[col].dropna().unique()
                ex = ", ".join(map(str, unique[:5]))
                buffer.write(f"- {col} ({dtype}): examples=[{ex}]\n")
            else:
                buffer.write(f"- {col} ({dtype})\n")
        return buffer.getvalue()

    def generate_code_with_llm(self, df: pd.DataFrame, prompt: str) -> str:
        if not self.llm_client:
             return None
        
        context = self.get_df_context(df)
        system_prompt = textwrap.dedent(f"""
        You are an expert Data Analyst. Write Python code to analyze the dataframe `df`.
        
        ### Data Context
        {context}
        
        ### Rules
        1. Use `df` which is already loaded.
        2. Use `pandas` as `pd`, `matplotlib.pyplot` as `plt`.
        3. Assign text/dataframe outputs to `result`.
        4. If plotting, create the figure but do NOT show it. Assign `result_img_path = None`.
        5. Return ONLY executable Python code inside ```python``` block.
        
        ### Request
        {prompt}
        """)
        
        resp = self.llm_client.ask_chat(system_prompt)
        
        if "```python" in resp:
            return resp.split("```python")[1].split("```")[0].strip()
        if "```" in resp:
            return resp.split("```")[1].split("```")[0].strip()
        return resp

    def run_code_safely(self, df: pd.DataFrame, code: str):
        # Prepare execution environment
        exec_globals = {"pd": pd, "np": np, "plt": plt, "df": df, "result": None, "result_img_path": None}
        exec_locals = {}
        
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, exec_globals, exec_locals)
        except Exception:
            return {"type": "error", "error": traceback.format_exc()}
            
        printed = buf.getvalue().strip()
        
        # Check visuals
        figs = plt.get_fignums()
        if figs:
            out_file = Path(tempfile.gettempdir()) / f"plot_{uuid.uuid4().hex}.png"
            plt.savefig(out_file, bbox_inches='tight')
            plt.close('all')
            return {"type": "image", "path": str(out_file), "logs": printed}
            
        # Check result var
        res = exec_locals.get("result", exec_globals.get("result"))
        if res is not None:
            if isinstance(res, (pd.DataFrame, pd.Series)):
                return {"type": "dataframe", "df": res if isinstance(res, pd.DataFrame) else res.to_frame(), "logs": printed}
            return {"type": "text", "output": str(res), "logs": printed}
            
        return {"type": "text", "output": printed or "No output.", "logs": printed}
