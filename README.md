# AI Data Analyst Agent

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CrewAI](https://img.shields.io/badge/Orchestration-CrewAI--Ready-FF4B4B.svg)](https://www.crewai.com/)

An intelligent, full-stack data analytics platform that turns raw tabular datasets into interactive visual reports, data quality profiles, key business metrics, and actionable recommendations using natural language.

---

## Executive Summary

Organizations often struggle with data silos and the technical barrier required to write SQL or Pandas scripts for rapid exploratory data analysis (EDA). **AI Data Analyst Agent** bridges this gap: users upload tabular files (CSV, Excel, Parquet, JSON, TSV), ask questions in plain English, and immediately receive:
- **Executive Summaries & Actionable Recommendations**
- **Automated Data Quality Profiling** (missing value rates, duplicates, data types)
- **Calculated Metric Highlights** (totals, averages, changes)
- **Interactive Plotly Visualizations**
- **Optional Web Research Context** (integrated research pipeline)

Built with a modular Python/FastAPI backend, modern React frontend, containerized with Docker, and architected for multi-agent LLM orchestration (CrewAI + Google Gemini).

---

## System Architecture

```text
               User / Browser
                     │
                     ▼
       ┌───────────────────────────┐
       │   React 18 + Vite (SPA)   │  ── Deployed on Vercel
       └─────────────┬─────────────┘
                     │ HTTP / Multipart Form-Data
                     ▼
       ┌───────────────────────────┐
       │     FastAPI REST API      │  ── Containerized (Docker)
       └─────────────┬─────────────┘
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
┌──────────────┐          ┌─────────────────┐
│ Pandas Core  │          │ Agent Framework │
│  & Plotly    │          │ (CrewAI/Gemini) │
└──────┬───────┘          └────────┬────────┘
       │                           │
       ▼                           ▼
 Data Profiling, Metrics,    Web Research &
  Charts & Data Hygiene    Executive Synthesis
```

---

## Key Features & Capabilities

- **Multi-Format Tabular Ingestion**:
  - Supports `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls` (with sheet selection), `.parquet`, `.json`, and `.ndjson`.
  - Resilient encoding handling (`utf-8`, `utf-8-sig`, `latin-1`).
- **Comprehensive Data Quality Checks**:
  - Automatically flags row/column counts, duplicate records, data type classifications, and column-wise missing value percentages.
- **Natural Language Data Querying**:
  - Interprets questions like *"What are the most important trends?"* or *"Analyze sales by category"* to compute relevant aggregates, group-bys, and trends.
- **Dynamic Visualizations**:
  - Generates interactive, standalone Plotly charts embedded directly in the frontend dashboard.
- **Production-Ready & Containerized**:
  - Multi-stage Docker build utilizing [`uv`](https://github.com/astral-sh/uv) for ultra-fast, reproducible dependency installations and minimal final image size.
  - Environment-based configurable CORS handling.
  - Ready for cloud deployment across Vercel (frontend) and Render/Railway/Cloud Run (backend).

---

## Tech Stack & Skills Highlighted

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.12+, FastAPI, Uvicorn, Pandas, Plotly Express, Pydantic, PyArrow, OpenPyXL |
| **Agent / AI** | CrewAI-ready orchestration, Google Gemini API, Nimble Web Research integration |
| **Frontend** | React 18, Vite, Modern CSS (Responsive Grid & Flexbox, Card UI) |
| **DevOps & Tooling**| Docker (Multi-stage), `uv` package manager, JupyterLab, Git, Vercel |

---

## Project Structure

```text
data-analyst-agent/
├── src/
│   └── data_analyst/
│       ├── api.py              # FastAPI endpoints, CORS, static chart mounting
│       ├── config.py           # Directory & environment configurations
│       ├── crew.py             # Agent orchestration boundary (deterministic + LLM)
│       └── tools/
│           ├── data_tools.py   # Ingestion, profiling, metrics calculation, Plotly engine
│           └── web_tools.py    # External context & web research module
├── frontend/                   # React + Vite Single Page Application
│   ├── src/
│   │   ├── main.jsx            # Interactive UI with upload, query, & metric dashboard
│   │   └── styles.css          # Modern dark-mode responsive styling
│   ├── index.html
│   └── package.json
├── data/                       # Ingestion directory (with sample datasets)
├── notebooks/                  # Validation notebooks & edge-case testing
├── Dockerfile                  # Multi-stage container build for production
├── .dockerignore               # Optimized Docker build context exclusions
├── .gitignore                  # Full-stack git ignore configuration
├── pyproject.toml              # Modern Python packaging with uv/setuptools
└── uv.lock                     # Deterministic dependency lockfile
```

---

## Quickstart Guide

### Prerequisites
- Python 3.12+ (or [`uv`](https://docs.astral.sh/uv/))
- Node.js 18+ and npm
- Docker (optional, for container runs)

### 1. Environment Setup

Clone the repository and copy the example environment configuration:
```bash
git clone https://github.com/<your-username>/data-analyst-agent.git
cd data-analyst-agent
cp .env.example .env
```
*(Optional: Provide `GEMINI_API_KEY` in `.env` if enabling LLM research agents).*

### 2. Run the Backend (FastAPI)

Using [`uv`](https://docs.astral.sh/uv/) (recommended):
```bash
uv sync
uv run uvicorn data_analyst.api:app --reload --port 8000
```
- API Health Check: `http://localhost:8000/health`
- Interactive API Docs (Swagger): `http://localhost:8000/docs`

### 3. Run the Frontend (React / Vite)

```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser. Upload `data/sample.csv` (or any custom tabular dataset) to test!

---

## Docker & Production Deployment

### Run Locally with Docker

```bash
# Build the container image
docker build -t data-analyst-agent .

# Run the container
docker run -p 8000:8000 --env-file .env data-analyst-agent
```

### Production Cloud Architecture
- **Backend**: Deploy the Docker container on **Render**, **Railway**, **Fly.io**, or **Google Cloud Run**. Configure `ALLOWED_ORIGINS` to accept requests from your frontend URL.
- **Frontend**: Deploy `frontend/` on **Vercel** with the environment variable `VITE_API_URL=https://<your-backend-domain>`.

---

## Edge Case Testing & Data Validation

The project includes pre-configured validation suites for real-world messy data scenarios (missing values, irregular delimiters, multi-sheet Excel files, date parsing):

```bash
uv sync --extra notebook
uv run jupyter lab
```
Navigate to `notebooks/edge_case_validation.ipynb` to inspect the validation runs across edge cases.

---

## Future Roadmap
- [ ] Direct database connectors (PostgreSQL, Snowflake, BigQuery).
- [ ] SSE / WebSocket streaming for real-time agent thought steps.
- [ ] Conversational follow-ups with in-memory chat session history.
