# Enterprise AI Data Analyst

## Overview

An enterprise-grade AI Data Analyst platform that enables business users
to query structured data using natural language, generate safe SQL,
visualize insights, create reports, and receive business recommendations
powered by Gemini and agentic workflows.

## Objectives

-   Convert natural language into SQL.
-   Execute safe SQL against PostgreSQL.
-   Generate dashboards and business insights.
-   Support conversational analytics.
-   Export reports.
-   Demonstrate production-grade architecture.

## Tech Stack

  Layer             Technology
  ----------------- -----------------------------------------------------
  Frontend          Next.js, React, TypeScript, Tailwind CSS, shadcn/ui
  Backend           FastAPI
  LLM               Gemini 2.5 Pro / Flash
  Agent Framework   LangGraph
  Database          PostgreSQL
  ORM               SQLAlchemy
  Vector Database   Qdrant
  Embeddings        Gemini Embeddings
  Authentication    Clerk/Auth.js
  Charts            Apache ECharts
  Queue             Redis + Celery
  Monitoring        Langfuse
  Packaging         uv
  Deployment        Docker

## Architecture

``` text
User
  │
Next.js Dashboard
  │
FastAPI
  │
LangGraph Agents
  ├── Intent Agent
  ├── SQL Generator
  ├── SQL Validator
  ├── Execution Agent
  ├── Visualization Agent
  ├── Insight Agent
  └── Report Agent
  │
PostgreSQL + Qdrant
  │
Gemini API
```

## Core Features

1.  Natural Language → SQL
2.  Safe SQL validation
3.  Interactive dashboards
4.  Automatic chart selection
5.  Executive business insights
6.  KPI generation
7.  Report export (PDF/Markdown/PPT)
8.  Conversational memory
9.  Data quality analysis
10. Semantic search over data dictionary

## Database Schema

### customers

-   customer_id
-   customer_name
-   country
-   credit_limit
-   risk_score

### invoices

-   invoice_id
-   customer_id
-   invoice_date
-   due_date
-   amount
-   status

### payments

-   payment_id
-   invoice_id
-   payment_date
-   payment_amount

## LangGraph Workflow

``` text
User Query
    ↓
Intent Detection
    ↓
Schema Retrieval
    ↓
SQL Generation
    ↓
SQL Validation
    ↓
Execute Query
    ↓
Visualization
    ↓
Business Insights
    ↓
Report Generation
```

## Folder Structure

``` text
enterprise-ai-data-analyst/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   ├── api/
│   │   ├── database/
│   │   ├── prompts/
│   │   ├── services/
│   │   └── main.py
│   ├── tests/
│   └── pyproject.toml
├── frontend/
├── docker/
├── docs/
└── README.md
```

## Development Roadmap

### Phase 1

-   Backend
-   PostgreSQL
-   Gemini integration
-   NL-to-SQL

### Phase 2

-   Frontend
-   Charts
-   Authentication

### Phase 3

-   LangGraph multi-agent workflow
-   SQL validation
-   Conversational memory

### Phase 4

-   Vector search
-   Report generation
-   Monitoring
-   Docker deployment

## Resume Highlights

-   Enterprise AI analytics platform using Gemini, FastAPI, LangGraph,
    PostgreSQL, and Next.js.
-   Multi-agent workflow for SQL generation, validation, visualization,
    and reporting.
-   Business-focused insights with safe query execution and
    conversational analytics.

## Future Enhancements

-   Role-based access control
-   Multi-database connectors
-   Scheduled analytics
-   Slack/Microsoft Teams integration
-   Forecasting models
-   Voice analytics
-   MCP server integration
-   BI tool connectors (Power BI/Tableau)
