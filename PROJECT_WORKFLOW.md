# Personal AI Data Analyst - Project Workflow

This document outlines the architectural workflow and data flow of the "Personal AI Data Analyst" application. The project is designed to automate data analysis tasks using a combination of deterministic logic and Large Language Models (LLMs).

## 1. High-Level Architecture

The application is built using **Streamlit** for the frontend and **Python** for the backend logic. It follows a modular design where the core analysis logic is separated from the UI.

### Key Components:
- **`app.py` / `main.py`**: The entry point for the Streamlit application. It handles user interaction, file uploads, and displays results.
- **`analyst.py`**: A utility module containing the core business logic:
  - Data loading and type detection.
  - Prompt suggestion generation.
  - Prompt-to-code conversion (Deterministic & LLM).
  - Safe code execution environment.
- **Local LLM (Ollama)**: An optional external service used to generate Python code for custom, complex prompts that don't match predefined templates.

## 2. Detailed Workflow Steps

The following steps describe the end-to-end flow of a user session:

### Step 1: Initialization (`app.py`)
- The app initializes the Streamlit page layout.
- A sidebar is rendered to allow configuration:
  - **Use local LLM**: A checkbox to enable/disable the integration with Ollama (default: `False`).
  - **LLM Model**: Text input to specify the model name (default: `llama3.1`).

### Step 2: Data Ingestion (`analyst.load_data`)
- The user uploads a file (CSV, Excel, or JSON).
- The `load_data` function in `analyst.py` handles the file:
  - Determines the file format based on extension or content inspection.
  - Reads the data into a **pandas DataFrame (`df`)**.
- A preview of the first 100 rows is displayed to the user.

### Step 3: Automated Suggestions (`analyst.suggest_prompts`)
- The `_detect_column_types` helper analyzes the DataFrame to match columns to types (Numeric, Datetime, Categorical).
- Based on these types, `suggest_prompts` generates a list of actionable strings, such as:
  - "Summarize the dataset"
  - "Create a histogram of [Numeric Column]"
  - "Show top 10 counts for [Categorical Column]"
- These suggestions are displayed in a dropdown menu.

### Step 4: Prompt Selection & Customization
- The user can either:
  1. Select a generic suggestion from the list.
  2. Write a custom natural language prompt (e.g., "filtering for sales > 500").
- The app reconciles these into a single `final_prompt`.

### Step 5: Interpretation (Prompt -> Code)
When "Run analysis" is clicked, the app determines how to generate the analysis code:

**A. Deterministic Path (`analyst.prompt_to_code`)**:
- The system attempts to match the `final_prompt` against regex patterns for known tasks (e.g., summaries, plots, sorting).
- **If matched**: It returns a robust, pre-written Python code snippet customized for the specific column names.

**B. LLM Path (`analyst.ask_llm`)**:
- **If no match** AND **LLM is enabled**:
  - The app constructs a system prompt describing the dataframe environment (`df`, `pd`, `plt`).
  - It sends the user's request to the local Ollama instance.
  - The LLM returns a Python code block.
- **If no match** AND **LLM is disabled**:
  - The system returns an error asking the user to enable the LLM or check their prompt.

### Step 6: Code Execution (`analyst.run_code`)
- The generated code string is passed to `run_code`.
- **Sandboxing**: The code is executed using Python's `exec()` within a controlled namespace containing `df`, `pd`, `np`, and `plt`.
- **Output Capture**:
  - **Text**: Captures standard output (`print`) or the final value of a `result` variable.
  - **DataFrames**: If `result` is a DataFrame, it is captured for interactive display.
  - **Images**: If `matplotlib` figures are created, they are saved to a temporary file and the path is returned.

### Step 7: Result Presentation
- **Text**: Displayed as raw text.
- **Table**: Displayed as an interactive Streamlit dataframe (with CSV download option).
- **Chart**: Displayed as an image.

## 3. File Structure Analysis

| File | Type | Description |
|------|------|-------------|
| `app.py` | UI | Main application dashboard (requires `src` in path). |
| `src/data_analysis/analyst.py` | Logic | Core business logic, moved to package structure. |
| `Dockerfile` | Ops | Container definition for the application. |
| `Makefile` | Ops | Automation commands. |
| `README.md` | Doc | Project overview, problem statement, and tech stack. |

## 4. Dependencies
- **Core**: `streamlit`, `pandas`, `numpy`, `matplotlib`, `scipy`, `openpyxl`
- **Optional**: `ollama` (external CLI tool)
