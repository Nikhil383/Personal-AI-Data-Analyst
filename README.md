# AI Data Analyst

An intelligent data analysis application powered by **LangGraph**, **SQL (DuckDB)** and the **Google Gemini API**. Ask questions about your data in natural language and get instant insights with visualizations.

## Features

- **Data Loading** - Support for CSV and Excel files
- **Natural Language Queries** - Ask questions about your data in plain English
- **Text-to-SQL with LangGraph** - Gemini converts questions into SQL, executed safely in DuckDB
- **Natural Language Answers** - Gemini summarizes query results into clear, concise answers
- **No Info Available Fallback** - If a question can't be answered from the data, the app honestly says "No info available"
- **Visualizations** - Auto-generate charts (histogram, bar, scatter, line, box, pie, correlation heatmap)
- **AI-Powered Insights** - Get intelligent analysis and suggestions
- **Modern Dark UI** - Beautiful, responsive interface

## How It Works

The app uses a **LangGraph state machine** to orchestrate the text-to-SQL workflow:

```
User Question → LangGraph → Gemini (generate SQL) → DuckDB (execute SQL) → Gemini (generate answer)
```

1. **Generate SQL** - Gemini converts your question into a safe `SELECT` query against your data's schema
2. **Execute** - The SQL runs against your data in an in-memory DuckDB engine (read-only, no mutations allowed)
3. **Generate Answer** - Gemini summarizes the query results in natural language
4. **No Info Available** - If SQL generation fails, results are empty, or the question can't be answered, the app replies "No info available" instead of hallucinating

### Graph Structure

```
START → generate_sql → execute_sql
                              ├─ (SQL error, retries left) → fix_sql → execute_sql
                              ├─ (no rows / retries exhausted) → no_info
                              └─ (success) → generate_answer → suggest_chart → END
```

SQL queries are validated before execution: only single `SELECT`/`WITH` statements are allowed, and dangerous keywords (`INSERT`, `DROP`, `DELETE`, etc.) are rejected.

## Prerequisites

- Python 3.11+
- uv (package manager)
- Google Gemini API Key

## Getting Your API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## Installation

### 1. Install uv (if not already installed)

```bash
# Windows
winget install astral-sh.uv

# Or via pip
pip install uv
```

### 2. Install Dependencies

```bash
# Create virtual environment and install
uv venv
uv sync

# Or install directly
uv pip install -e .
```

### 3. Configure Environment

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Run the Application

```bash
# Activate virtual environment (if using venv)
# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate

# Run Streamlit app
streamlit run src/ai_data_analyst/main.py
```

The app will open at `http://localhost:8501`

### Using the Application

1. **Upload Data** - Drag & drop or select a CSV/Excel file in the sidebar
2. **Preview Data** - View your data in the "Data Preview" tab
3. **Ask Questions** - Use the "Analysis" tab to ask questions in natural language
   - Examples: "What is the total sales?", "Show me top products by revenue", "What's the average profit by category?"
4. **Create Charts** - Use the "Visualizations" tab to generate various chart types

## Project Structure

```
ai-data-analyst/
├── .env.example          # Environment variables template
├── pyproject.toml       # Project configuration
├── data/                # Sample data directory
├── output/
│   └── charts/          # Generated charts
├── src/
│   └── ai_data_analyst/
│       ├── __init__.py
│       ├── config.py           # Configuration
│       ├── data_loader.py      # Data loading utilities
│       ├── analyzer.py         # Data analysis (legacy pandas ReAct)
│       ├── visualizer.py       # Chart generation
│       ├── chains/
│       │   ├── __init__.py
│       │   ├── analyst_chain.py           # Legacy LangChain pandas agent
│       │   ├── enhanced_analyst_chain.py  # Legacy enhanced ReAct chain
│       │   └── sql_analyst_chain.py       # New LangGraph + SQL chain
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── state.py            # LangGraph state schema
│       │   ├── sql_engine.py       # DuckDB SQL engine
│       │   ├── nodes.py            # Graph nodes (generate_sql, execute_sql, ...)
│       │   └── analyst_graph.py    # Graph builder & compiler
│       └── main.py              # Streamlit app (uses LangGraph + SQL)
└── tests/               # Test files
```

## Sample Questions to Try

- "What is the total sales across all products?"
- "Show the average profit by category"
- "What are the top 5 products by revenue?"
- "What is the correlation between sales and profit?"
- "Show me the distribution of customer ratings"
- "What is the average discount by region?"
- "What is the weather forecast?" (returns "No info available")

## Technology Stack

- **Python 3.11+**
- **uv** - Package manager
- **Streamlit** - Web framework
- **LangGraph** - Stateful agent orchestration
- **DuckDB** - In-process SQL engine
- **LangChain** - LLM integration
- **Google Gemini API** - LLM (text-to-SQL + answer generation)
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization

## Troubleshooting

### API Key Error
If you get an API key error, make sure:
1. Your `.env` file exists in the project root
2. The `GOOGLE_API_KEY` variable is set correctly
3. You have internet connectivity

Without an API key, the app still runs, but all queries return "No info available".

### Import Errors
Ensure all dependencies are installed:
```bash
uv pip install -r pyproject.toml
```

### Large Files
For files larger than 100MB, consider:
- Sampling the data first
- Removing unnecessary columns

## License

MIT
