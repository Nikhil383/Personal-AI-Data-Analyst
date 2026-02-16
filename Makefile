# Makefile for Personal AI Data Analyst

IMAGE_NAME = personal-ai-data-analyst

.PHONY: help install run docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies using uv"
	@echo "  make run           - Run the Streamlit app"
	@echo "  make docker-build  - Build the Docker image"
	@echo "  make docker-run    - Run the Docker container"
	@echo "  make clean         - Remove temporary files"

install:
	uv sync

run:
	uv run streamlit run frontend/streamlit_app.py

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p 8501:8501 $(IMAGE_NAME)

clean:
	rm -rf __pycache__
	rm -rf .venv
	rm -rf *.egg-info
