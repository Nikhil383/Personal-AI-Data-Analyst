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
from data_analysis.analyst import load_data, suggest_prompts, prompt_to_code, run_code, generate_analysis_code, get_df_context, generate_initial_report
from data_analysis.vision import analyze_plot

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
        
        # Generate initial report (Auto-Analysis)
        # We can do this async in a real app, but for now we wait (it's text only)
        initial_report = generate_initial_report(df)
        
        return jsonify({
            'message': 'File uploaded', 
            'suggestions': suggestions,
            'initial_report': initial_report
        })
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
            # New centralized way
            code = generate_analysis_code(df, prompt)
            
            if code.startswith("[LLM"):
               return jsonify({'error': code})

        # 3. Run code
        res = run_code(df, code)
        
        response_data = {'type': 'text', 'output': ''}
        
        if res['type'] == 'text':
            response_data = {'type': 'text', 'output': res['output']}
        
        elif res['type'] == 'dataframe':
            df_res = res['df']
            response_data = {
                'type': 'dataframe', 
                'data': df_res.head(100).to_dict(orient='records'),
                'columns': df_res.columns.tolist(),
                'output': "Here is the data you requested:"
            }
            
        elif res['type'] == 'image':
            src_path = res['path']
            filename = os.path.basename(src_path)
            dst_path = os.path.join(STATIC_PLOTS_FOLDER, filename)
            shutil.copy(src_path, dst_path)
            url = f"/static/plots/{filename}"
            
            # Vision Analysis
            vision_insight = analyze_plot(dst_path)
            
            response_data = {
                'type': 'image', 
                'url': url,
                'output': vision_insight  # Add the vision analysis as text output
            }
            
        else:
            return jsonify({'error': 'Unknown result type'})

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f"Processing failed: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)