# Makefile for Personal AI Data Analyst

IMAGE_NAME = personal-ai-data-analyst
PORT = 8501

.PHONY: help install run docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies (requires uv)"
	@echo "  make run           - Run the Streamlit app locally"
	@echo "  make docker-build  - Build the Docker image"
	@echo "  make docker-run    - Run the Docker container"
	@echo "  make clean         - Remove temporary files"

install:
	uv venv
	@echo "Activate venv with: . .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)"
	uv pip install -e .

run:
	streamlit run app.py

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p $(PORT):$(PORT) $(IMAGE_NAME)

clean:
	rm -rf __pycache__
	rm -rf .venv
	rm -rf *.egg-info
