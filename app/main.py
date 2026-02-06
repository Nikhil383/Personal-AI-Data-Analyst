from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import pandas as pd
from typing import Optional

# Import local modules
from app.llm.gemini_client import GeminiClient
from app.analytics.pandas_analyzer import PandasAnalyzer

app = FastAPI(title="AI Data Analyst API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons (simple dependency injection)
gemini_client = GeminiClient()
analyzer = PandasAnalyzer(llm_client=gemini_client)

# In-memory storage for demo purposes (replace with Redis/DB in prod)
FILES_DIR = "data/raw"
os.makedirs(FILES_DIR, exist_ok=True)
ACTIVE_DF = None
ACTIVE_FILE_PATH = None

class AnalysisRequest(BaseModel):
    prompt: str

@app.get("/")
def health_check():
    return {"status": "ok", "version": "2.0"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global ACTIVE_DF, ACTIVE_FILE_PATH
    try:
        file_path = os.path.join(FILES_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load df
        ACTIVE_DF = analyzer.load_data(file_path)
        ACTIVE_FILE_PATH = file_path
        
        # Generate initial insights
        context = analyzer.get_df_context(ACTIVE_DF)
        initial_insight = gemini_client.ask_chat(f"Analyze this dataset schema and provide 3 key initial observations:\n{context}")
        
        return {
            "filename": file.filename,
            "rows": len(ACTIVE_DF),
            "columns": list(ACTIVE_DF.columns),
            "initial_insight": initial_insight
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_data(req: AnalysisRequest):
    global ACTIVE_DF
    if ACTIVE_DF is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    
    try:
        # 1. Generate Code
        code = analyzer.generate_code_with_llm(ACTIVE_DF, req.prompt)
        if not code:
            return {"type": "error", "output": "Could not generate code."}
            
        # 2. Execute Code
        result = analyzer.run_code_safely(ACTIVE_DF, code)
        
        # 3. Vision Analysis (if image)
        if result['type'] == 'image':
            vision_text = gemini_client.analyze_image(result['path'])
            result['vision_analysis'] = vision_text
            
        return result
        
    except Exception as e:
        return {"type": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
