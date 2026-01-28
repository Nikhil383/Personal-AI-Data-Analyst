# Personal AI Data Analyst

Personal AI Data Analyst is a web application that allows users to analyze their data using AI. It is built using Streamlit and can be run locally.

## Problem Statement

Analyzing data typically requires technical knowledge of Python, statistics, and visualization libraries. Many non-technical users struggle with:

- Understanding data types

- Writing analysis code

- Detecting patterns or anomalies

- Performing quick visual or statistical exploration

Personal AI Data Analyst solves this by acting as an automated “AI Data Scientist” that understands natural language and translates it into reproducible, executable Python code—running entirely locally.

## Tech Stack

- Streamlit
- Python
- pandas
- matplotlib
- ollama
- uv
- Docker (new)

## Learnings & Challenges

### Learnings

- How to integrate LLMs with deterministic analysis workflows
- Building a safe execution environment for AI-generated code
- Converting ambiguous natural language into reproducible Python logic.

### Challenges

- Ensuring safety while executing LLM-generated code

- Handling corrupted or large dataset files

- Maintaining accuracy in column-type detection

- Keeping the UI responsive even with large datasets

## Future Improvements

- UI will remove model selection option (auto-detect model)
- Support for SQL queries
- Plugin system for advanced analysis modules
- Export entire analysis as a PDF report
- Add caching layer for faster re-runs

## License

MIT
