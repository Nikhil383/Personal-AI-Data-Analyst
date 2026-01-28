import os
import sys
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, send_file
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_analysis.analyst import load_data, suggest_prompts, prompt_to_code, run_code, ask_llm, get_df_context

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads'
STATIC_PLOTS_FOLDER = os.path.join('static', 'plots')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_PLOTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Store filename in session
    session['filename'] = filename
    session['filepath'] = filepath
    
    try:
        # Load just to get suggestions
        df = load_data(filepath)
        suggestions = suggest_prompts(df)
        # Convert df info to JSON compatible format if needed, mostly we just need suggestions now
        return jsonify({'message': 'File uploaded', 'suggestions': suggestions})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'No file uploaded or file lost provided'})
    
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'})

    try:
        df = load_data(filepath)
        
        # 1. Try deterministic code generation
        code = prompt_to_code(prompt, df)
        
        # 2. If no code, ask LLM
        if not code:
            context = get_df_context(df)
            system = (
                "You are a helpful data analyst and will respond with Python code only.\n"
                "You must return code inside a ```python ... ``` block. The DataFrame is named `df`.\n"
                "Use pandas for data manipulation and matplotlib for charts. Do not import heavy libs.\n"
                "If returning a chart, produce matplotlib code that draws the figure (no show()) and nothing else.\n"
                "Here is the dataset schema:\n"
                f"{context}"
            )
            full_prompt = system + "\n# User prompt: " + prompt
            
            # Use configured model from env (handled in analyst.py defaults or passed here)
            # We rely on analyst.py reading env vars if we pass None, but let's be explicit if needed.
            # actually analyst.py ask_llm reads env if api_key is None.
            llm_out = ask_llm(full_prompt)
            
            if llm_out.startswith("[LLM"):
                return jsonify({'error': llm_out})
                
            if "```python" in llm_out:
                code = llm_out.split("```python")[1].split("```")[0]
            else:
                return jsonify({'error': "LLM did not return Python code", 'debug': llm_out})

        # 3. Run code
        res = run_code(df, code)
        
        if res['type'] == 'text':
            return jsonify({'type': 'text', 'output': res['output']})
        
        elif res['type'] == 'dataframe':
            df_res = res['df']
            # Convert to dict for JSON
            # limit output size?
            return jsonify({
                'type': 'dataframe', 
                'data': df_res.head(100).to_dict(orient='records'),
                'columns': df_res.columns.tolist()
            })
            
        elif res['type'] == 'image':
            # Move image to static
            src_path = res['path']
            filename = os.path.basename(src_path)
            dst_path = os.path.join(STATIC_PLOTS_FOLDER, filename)
            shutil.copy(src_path, dst_path)
            # URL for frontend
            url = f"/static/plots/{filename}"
            return jsonify({'type': 'image', 'url': url})
            
        else:
            return jsonify({'error': 'Unknown result type'})

    except Exception as e:
        return jsonify({'error': f"Processing failed: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)