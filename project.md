# PROJECT.md
# Enterprise AI Data Analyst v2 (Production-Grade Agentic AI System)

## Goal

Transform the existing AI Data Analyst application into a production-grade enterprise AI system that demonstrates modern AI engineering practices.

The project should showcase:

- LangGraph Agentic Workflows
- FastAPI Backend
- Streamlit Frontend
- Gemini API
- PostgreSQL
- Docker
- CI/CD
- RAG for Business Metadata
- Observability
- Production Architecture

This project should be interview-ready for AI Engineer, GenAI Engineer, and Forward Deployed Engineer roles.

---

# Existing Features

Current project already supports:

- CSV / Excel Upload
- Natural Language Querying
- Gemini API
- LangChain
- Data Analysis
- Chart Generation
- Streamlit UI

---

# Target Architecture

```
                    User

                      │

              Streamlit Frontend

                      │

                  REST API

                 (FastAPI)

                      │

               LangGraph Workflow

                      │

 ┌─────────────────────────────────────────────┐
 │                                             │
 │ Intent Detection Agent                      │
 │ Schema Retrieval Agent                      │
 │ SQL Generation Agent                        │
 │ SQL Validation Agent                        │
 │ Query Execution Agent                       │
 │ Visualization Agent                         │
 │ Business Insight Agent                      │
 │ Recommendation Agent                        │
 └─────────────────────────────────────────────┘

                      │

          PostgreSQL + Pandas

                      │

            Charts + Insights

                      │

                Final Response
```

---

# Tech Stack

## AI

- Gemini 2.5 Flash
- LangGraph
- LangChain
- Google AI SDK

---

## Backend

- FastAPI
- Pydantic
- Uvicorn

---

## Database

- PostgreSQL
- SQLAlchemy

---

## Frontend

- Streamlit

---

## DevOps

- Docker
- Docker Compose
- GitHub Actions

---

## Observability

- LangSmith

---

## Future

- Redis
- Celery
- AWS Lambda

---

# Folder Structure

```
enterprise-ai-data-analyst/

src/

    backend/

        api/

        routes/

        services/

        agents/

        graph/

        database/

        models/

        schemas/

        utils/

    frontend/

    shared/

docker/

tests/

docs/

```

---

# Project Roadmap

---

## Phase 1

Backend Separation

Current

```
Streamlit
↓

LLM
```

Target

```
Streamlit

↓

FastAPI

↓

LLM
```

Tasks

- Create FastAPI project
- Create REST endpoints
- Move business logic into backend
- Keep Streamlit as UI only

---

## Phase 2

Database Layer

Tasks

- Store uploaded datasets

- PostgreSQL integration

- SQLAlchemy ORM

- Data versioning

Endpoints

```
POST /upload

GET /datasets

DELETE /dataset

GET /columns
```

---

## Phase 3

LangGraph Migration

Replace LangChain chain with LangGraph.

Workflow

```
START

↓

Load Dataset

↓

Understand User Query

↓

Retrieve Dataset Metadata

↓

Generate SQL

↓

Validate SQL

↓

Execute Query

↓

Generate Visualization

↓

Generate Business Insight

↓

Generate Recommendations

↓

END
```

---

# LangGraph State

```python
class GraphState(TypedDict):

    user_query: str

    dataset_name: str

    dataframe: pd.DataFrame

    schema: dict

    generated_sql: str

    sql_result: Any

    visualization: dict

    insight: str

    recommendation: str

    retry_count: int

    error: str
```

---

# LangGraph Nodes

## Node 1

Dataset Loader

Responsibilities

- Load dataframe
- Validate dataset
- Store metadata

---

## Node 2

Intent Classifier

Determine

- Analysis

- Aggregation

- Trend

- Visualization

- Correlation

---

## Node 3

Schema Retrieval

Return

- columns

- datatype

- statistics

- null values

---

## Node 4

SQL Generator

Gemini generates SQL

Prompt includes

- schema

- column names

- user query

---

## Node 5

SQL Validator

Validate

- syntax

- missing columns

- invalid table

- SQL injection

If failed

↓

Retry SQL generation

---

## Node 6

Query Executor

Execute SQL

Return dataframe

---

## Node 7

Visualization Generator

Automatically choose

- Histogram

- Bar

- Pie

- Scatter

- Heatmap

- Line

- Boxplot

---

## Node 8

Insight Generator

Gemini explains

- trends

- anomalies

- business impact

---

## Node 9

Recommendation Agent

Example

Sales dropped

↓

Suggest

Increase promotions

Restock inventory

Check seasonal effects

---

# Conditional Edges

```
Generate SQL

↓

Validation

↓

Valid?

YES

↓

Execute

NO

↓

Retry
```

Retry

Maximum

3

After

3 retries

↓

Return error

---

# REST APIs

Dataset

```
POST /upload
```

Analysis

```
POST /analyze
```

Visualization

```
POST /visualize
```

Metadata

```
GET /schema
```

Health

```
GET /health
```

---

# Docker

Containers

```
Frontend

Backend

PostgreSQL

LangSmith
```

Use docker-compose.

---

# CI/CD

GitHub Actions

Pipeline

```
Lint

↓

Unit Tests

↓

Docker Build

↓

Deployment
```

---

# Logging

Use

- Python logging

- LangSmith tracing

Log

- Prompt

- Response

- Token usage

- Errors

---

# Error Handling

Handle

- Invalid dataset

- Missing columns

- Empty dataframe

- Invalid SQL

- API timeout

- Gemini rate limit

- Database failure

---

# Security

- Environment variables

- API key protection

- Input validation

- SQL injection prevention

- File validation

- File size limits

---

# Stretch Goals

## Multi Dataset Analysis

User

```
Compare Sales.csv with Inventory.csv
```

---

## Memory

Remember previous analysis.

---

## RAG

Store

- dataset documentation

- business glossary

- KPI definitions

Retrieve before answering.

---

## Multi-Agent Collaboration

Planner Agent

↓

SQL Agent

↓

Visualization Agent

↓

Business Analyst Agent

↓

Reviewer Agent

---

## Authentication

JWT

User login

Dataset ownership

---

## Export

Generate

- PDF reports

- Excel reports

- PowerPoint summaries

---

# Interview Talking Points

After completion, be able to confidently explain:

- Why LangGraph instead of LangChain?
- Why FastAPI instead of Streamlit-only architecture?
- How does the SQL validation node work?
- What information is stored in GraphState?
- How do conditional edges improve reliability?
- How is hallucination reduced?
- How does the retry mechanism work?
- Why PostgreSQL instead of Pandas alone?
- How would you scale the system for 10,000 concurrent users?
- How would you deploy this on AWS?

---

# Success Criteria

The project is complete when it demonstrates:

- Production-ready architecture
- Agentic workflows using LangGraph
- REST APIs with FastAPI
- PostgreSQL integration
- SQL validation and retry logic
- Dockerized deployment
- CI/CD pipeline
- LangSmith observability
- Comprehensive documentation
- Defensible architecture suitable for senior technical interviews