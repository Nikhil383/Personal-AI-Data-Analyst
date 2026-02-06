# Personal AI Data Analyst (V2 Production Architecture)

This project has been upgraded to a modular production-ready architecture.

## Architecture
- **Backend**: FastAPI (`app/main.py`)
- **Frontend**: Streamlit (`frontend/streamlit_app.py`)
- **Core Logic**:
  - `app/analytics/pandas_analyzer.py`: Dataframe operations
  - `app/llm/gemini_client.py`: LLM & Vision integration

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # OR
   pip install fastapi uvicorn streamlit pandas matplotlib langchain-google-genai python-dotenv python-multipart
   ```

2. **Environment Variables**
   Ensure `.env` contains:
   ```
   GOOGLE_API_KEY=your_key_here
   GEMINI_MODEL=gemini-pro
   ```

## Running the App

You need to run **both** the backend and frontend.

**1. Start Backend API**
```bash
uvicorn app.main:app --reload
```
API will allow uploads at `http://localhost:8000`.

**2. Start Frontend UI** (in a new terminal)
```bash
streamlit run frontend/streamlit_app.py
```
The UI will open in your browser (usually `http://localhost:8501`).

## Directory Structure
(As requested)
- `app/`: Source code for API and Analytics logic.
- `frontend/`: Streamlit UI code.
- `data/`: Data storage.
- `notebooks/`: Jupyter notebooks.
