# AI Data Analyst Agent

Upload a tabular dataset, ask a business question, and receive local data-quality checks, metrics, trends, recommendations, and Plotly charts.

Supported uploads: CSV, TSV, delimited text, Excel (`.xlsx`, `.xls`), JSON, NDJSON, and Parquet. For Excel workbooks, the first worksheet is analyzed by default; the API also accepts an optional `sheet_name` form field.

## Run the API

```bash
uv sync
uv run uvicorn data_analyst.api:app --reload
```

## Run the web app

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, select a dataset, and submit a question. The included `data/sample.csv` is a quick test dataset.

Copy `.env.example` to `.env` to configure Gemini and Nimble credentials for future agent/research extensions. The present implementation keeps core CSV analysis fully local and deterministic.

## Validate an Excel sample in a notebook

Place an `.xlsx` file in `data_sample/`, then install the optional notebook tools and start Jupyter:

```bash
uv sync --extra notebook
uv run jupyter lab
```

Run [notebooks/edge_case_validation.ipynb](notebooks/edge_case_validation.ipynb). It loads the supplied Excel file, exercises the project analysis logic, and checks common CSV edge cases.
