import os
import uuid
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import UPLOAD_DIR
from app.database.models import get_db, Base, engine, Customer, Invoice, Payment, Dataset
from app.agents.workflow import workflow
import random

app = FastAPI(title="Enterprise AI Data Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    intent: str
    sql_query: Optional[str]
    sql_valid: Optional[bool]
    sql_error: Optional[str]
    query_results: Optional[List[Dict[str, Any]]]
    visualization: Optional[str]
    insights: Optional[str]

@app.post("/api/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest):
    try:
        inputs = {"query": payload.query}
        output = workflow.invoke(inputs)
        return QueryResponse(
            query=payload.query,
            intent=output.get("intent", "sql"),
            sql_query=output.get("sql_query"),
            sql_valid=output.get("sql_valid"),
            sql_error=output.get("sql_error"),
            query_results=output.get("query_results"),
            visualization=output.get("visualization_code"),
            insights=output.get("insights")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/db/init")
def initialize_database(db: Session = Depends(get_db)):
    """Creates tables and populates sample records."""
    try:
        Base.metadata.create_all(bind=engine)
        
        # Check if we already have data
        if db.query(Customer).count() > 0:
            return {"status": "success", "message": "Database already initialized."}
            
        # Seed Customers
        countries = ["USA", "Canada", "UK", "Germany", "France", "Japan", "Australia"]
        customers = []
        for i in range(1, 21):
            cust = Customer(
                customer_name=f"Enterprise Client {i}",
                country=random.choice(countries),
                credit_limit=round(random.uniform(50000, 500000), 2),
                risk_score=round(random.uniform(0.1, 0.9), 2)
            )
            db.add(cust)
            customers.append(cust)
            
        db.commit()
        
        # Seed Invoices and Payments
        statuses = ["paid", "unpaid", "overdue"]
        for cust in customers:
            for j in range(random.randint(2, 6)):
                inv_date = date.today() - timedelta(days=random.randint(10, 180))
                due_date = inv_date + timedelta(days=30)
                amount = round(random.uniform(1000, 25000), 2)
                status = random.choice(statuses)
                
                invoice = Invoice(
                    customer_id=cust.customer_id,
                    invoice_date=inv_date,
                    due_date=due_date,
                    amount=amount,
                    status=status
                )
                db.add(invoice)
                db.commit()
                
                if status == "paid":
                    payment = Payment(
                        invoice_id=invoice.invoice_id,
                        payment_date=inv_date + timedelta(days=random.randint(1, 28)),
                        payment_amount=amount
                    )
                    db.add(payment)
                    db.commit()
                    
        return {"status": "success", "message": "Database tables created and seeded successfully."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    allowed_ext = {"csv", "xlsx"}
    filename = Path(file.filename).name
    extension = filename.split(".")[-1].lower()
    if extension not in allowed_ext:
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported.")

    unique_name = f"{uuid.uuid4().hex}_{filename}"
    file_path = UPLOAD_DIR / unique_name

    try:
        contents = await file.read()
        file_path.write_bytes(contents)

        if extension == "csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        dataset = Dataset(
            name=filename,
            filename=unique_name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=",".join([str(col) for col in df.columns]),
            file_path=str(file_path)
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        return {"status": "success", "dataset": dataset.to_dict()}
    except Exception as e:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    return {"datasets": [ds.to_dict() for ds in datasets]}

@app.get("/api/dataset/{dataset_id}/columns")
def get_dataset_columns(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"dataset_id": dataset.dataset_id, "columns": dataset.columns.split(",")}

@app.delete("/api/dataset/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        if dataset.file_path and Path(dataset.file_path).exists():
            Path(dataset.file_path).unlink()
        db.delete(dataset)
        db.commit()
        return {"status": "success", "message": "Dataset deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/schema")
def get_schema_info():
    """Returns database schema information."""
    return {
        "tables": {
            "customers": ["customer_id", "customer_name", "country", "credit_limit", "risk_score"],
            "invoices": ["invoice_id", "customer_id", "invoice_date", "due_date", "amount", "status"],
            "payments": ["payment_id", "invoice_id", "payment_date", "payment_amount"]
        }
    }
