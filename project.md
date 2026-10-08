# AI Data Analyst Agent

## 1. Project Overview

An AI-powered data analyst that allows users to upload CSV datasets and ask business questions in natural language.

The system uses **CrewAI** to coordinate specialized agents for data analysis, external research, and report generation.

## 2. Objectives

- Analyze CSV datasets using natural language.
- Automatically inspect data quality.
- Calculate relevant metrics and trends.
- Generate business insights and recommendations.
- Use web research when external context is required.
- Provide results through a simple web interface.

## 3. Tech Stack

### Backend

- Python 3.12+
- CrewAI
- Gemini
- Pandas
- Plotly
- FastAPI
- Pydantic
- python-dotenv
- uv

### Web Research

- Nimble Web Search Agent

### Frontend

- React
- Vite
- JavaScript
- CSS

## 4. Agent Architecture

```text
                    User
                     |
                     v
               React Frontend
                     |
                     v
                FastAPI API
                     |
                     v
               CrewAI Crew
                     |
        +------------+------------+
        |            |            |
        v            v            v
 Data Analyst   Research Agent  Report Agent
        |            |            |
        v            v            v
     Pandas       Nimble       Final Report
        |            |
        +------------+
             |
             v
        Business Insights
```

## 5. Agents

### Data Analyst

Responsibilities:

- Inspect dataset structure.
- Detect missing values and duplicates.
- Calculate statistics.
- Analyze trends and relationships.
- Generate charts when required.

Tools:

- `analyze_csv`
- `query_csv`
- `save_chart`

### Research Analyst

Responsibilities:

- Determine whether external context is needed.
- Research current information using Nimble.
- Separate external facts from dataset findings.

Tool:

- `web_research`

### Report Analyst

Responsibilities:

- Combine analysis and research.
- Produce an executive summary.
- Highlight important metrics.
- Explain findings.
- Provide recommendations and caveats.

## 6. Data Flow

```text
CSV Upload
    ↓
Dataset Profiling
    ↓
Data Quality Analysis
    ↓
Metric Calculation
    ↓
Optional Web Research
    ↓
Insight Generation
    ↓
Final Analyst Report
```

## 7. Project Structure

```text
ai-data-analyst/
│
├── src/
│   └── data_analyst/
│       ├── agents.py
│       ├── tasks.py
│       ├── crew.py
│       ├── api.py
│       ├── main.py
│       ├── config.py
│       └── tools/
│           ├── data_tools.py
│           └── web_tools.py
│
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   └── package.json
│
├── data/
│   └── sample.csv
│
├── outputs/
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
└── project.md
```

## 8. Installation

Create the environment with `uv`:

```bash
uv sync
```

Create `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
NIMBLE_API_KEY=your_nimble_api_key
GEMINI_MODEL=gemini-3.8-flash
```

## 9. Run Backend

```bash
uv run uvicorn data_analyst.api:app --reload
```

API:

```text
http://localhost:8000
```

Health check:

```text
GET /health
```

Analysis endpoint:

```text
POST /analyze
```

## 10. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## 11. Example Query

Upload:

```text
sales.csv
```

Ask:

```text
Why did revenue decrease in Q3?
```

The agent should:

1. Inspect the dataset.
2. Check data quality.
3. Calculate revenue metrics.
4. Identify relevant trends.
5. Research external context if necessary.
6. Generate a final report.

## 12. Current Limitations

- CSV is currently the primary supported input.
- SQL database integration is not yet implemented.
- Charts are currently generated as Plotly HTML files.
- Authentication is not implemented.
- Large datasets require further optimization.

## 13. Future Improvements

### Phase 1

- PostgreSQL integration.
- SQL Analyst Agent.
- Excel support.
- More Plotly visualizations.

### Phase 2

- Dashboard generation.
- Streaming agent execution.
- Persistent analysis history.
- Authentication.

### Phase 3

- Multi-database support.
- Automated KPI monitoring.
- Scheduled reports.
- Agent memory.
- Production deployment with Docker.

## 14. Expected Outcome

The final application should behave like an AI business analyst:

```text
User Question
     ↓
Understand Intent
     ↓
Inspect Data
     ↓
Analyze
     ↓
Research
     ↓
Visualize
     ↓
Explain
     ↓
Recommend
```

The goal is to move from a simple chatbot to a practical **agentic data-analysis platform**.
