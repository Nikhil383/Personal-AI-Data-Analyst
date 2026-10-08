from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR, OUTPUT_DIR
from .crew import run_analysis
from .tools.data_tools import SUPPORTED_TABULAR_EXTENSIONS, load_tabular

import os

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
] or ["*"]

app = FastAPI(title="AI Data Analyst", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def root() -> dict:
    return {
        "service": "ai-data-analyst",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "message": "AI Data Analyst API is running.",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-data-analyst"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(...),
    research: bool = Form(False),
    sheet_name: str | None = Form(None),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if not file.filename or suffix not in SUPPORTED_TABULAR_EXTENSIONS:
        formats = ", ".join(sorted(ext.lstrip(".").upper() for ext in SUPPORTED_TABULAR_EXTENSIONS))
        raise HTTPException(400, f"Please upload a supported tabular file: {formats}.")
    if not question.strip():
        raise HTTPException(400, "Please enter a question about the data.")
    upload_path = DATA_DIR / f"{uuid4().hex}_{Path(file.filename).name}"
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        dataframe = load_tabular(upload_path, sheet_name=sheet_name or 0)
        if dataframe.empty:
            raise HTTPException(400, "The CSV file contains no rows.")
        return {
            "filename": file.filename,
            "file_type": suffix.lstrip("."),
            "sheet_name": sheet_name if suffix in {".xlsx", ".xls"} else None,
            "question": question,
            **run_analysis(dataframe, question, OUTPUT_DIR, research),
        }
    except UnicodeError as exc:
        raise HTTPException(400, "The CSV encoding could not be read.") from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(400, f"Could not analyze CSV: {exc}") from exc
    finally:
        await file.close()
