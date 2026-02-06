# Project Workflow - AI Data Analyst

This project follows a streamlined workflow to convert user data and natural language queries into executable Python code and visual insights.

## Workflow Overview

1.  **Data Loading & Validation**:
    *   User uploads a file (CSV, Excel).
    *   The system detects the file type and loads it into a `pandas.DataFrame`.
    *   Basic metadata (columns, types, missing values) is extracted.

2.  **Prompt Suggestion**:
    *   Based on column types (numeric, categorical, datetime), `suggest_prompts()` generates relevant starting questions for the user (e.g., "Show correlation matrix", "Top 10 items in category X").

3.  **Query Processing**:
    *   User inputs a natural language query (e.g., "What is the average price by month?").
    *   **Step 3a (Deterministic)**: The system checks `prompt_to_code()` for exact matches with pre-defined templates (fast, reliable).
    *   **Step 3b (LLM Generation)**: If no template matches, the system calls `generate_analysis_code()`, which sends the query + schema context to the LLM (Gemini) with a strict system prompt to output valid Python code.

4.  **Code Execution (Sandboxed)**:
    *   The generated code is executed via `run_code()`.
    *   Execution happens in a controlled namespace with access to `pd`, `plt`, `np`, and the loaded `df`.
    *   The system captures:
        *   Standard output (print statements).
        *   `result` variable (DataFrames, values).
        *   `result_img_path` or active matplotlib figures.

5.  **Result Visualization**:
    *   **Text/Data**: Returned as JSON/HTML tables.
    *   **Images**: Saved to `static/plots/` and displayed in the frontend.

## Key Components

*   `app.py`: Flask web server and API endpoints.
*   `src/data_analysis/analyst.py`: Core logic for data loading, deterministic generation, LLM interaction, and code execution.
*   `templates/index.html`: Frontend UI.

## Adding New Analysis Capabilities

To add specific hardcoded templates:
1.  Modify `prior_to_code` in `src/data_analysis/analyst.py` with regex for your query.

To improve general understanding:
1.  Update the `system_prompt` in `generate_analysis_code` in `src/data_analysis/analyst.py` to include new rules or library instructions.
