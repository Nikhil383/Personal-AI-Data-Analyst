# Implementation Plan - Production Restructure

## Goal
Restructure the 'Personal AI Data Analyst' project to a production-ready layout, adding Docker support and automation tools.

## 1. Directory Structure
- **Current**: Flat structure
- **Target**:
    ```
    .
    ├── src/
    │   └── data_analysis/
    │       ├── __init__.py
    │       └── analyst.py
    ├── tests/
    ├── app.py              # Entry point (Streamlit)
    ├── main.py             # Alternative entry point (keep or merge?)
    ├── Dockerfile          # NEW
    ├── Makefile            # NEW
    ├── pyproject.toml      # Update
    ├── README.md
    └── PROJECT_WORKFLOW.md
    ```

## 2. File Changes
- **`src/data_analysis/analyst.py`**: Moved from root.
- **`app.py`**: Update imports to use `data_analysis.analyst` or local path adjustment.
- **`pyproject.toml`**: Configure for `src` layout (standard python packaging).

## 3. New Files
- **`Dockerfile`**: Containerize the Streamlit application using `uv` for dependency management.
- **`Makefile`**: Commands for `install`, `run`, `lint`, `docker-build`, `docker-run`.

## 4. Verification
- Verify the app starts locally (`make run`).
- Verify the app builds and runs in Docker (`make docker-run`).
