"""AI Data Analyst - Configuration Module"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API key
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
    raise ValueError(
        "GOOGLE_API_KEY not found. Please set it in .env file. "
        "Get your API key from: https://aistudio.google.com/app/apikey"
    )

# LLM Configuration
GEMINI_MODEL = "gemini-2.5-flash"

# LLM Parameters
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.2
TOP_P = 0.95
TOP_K = 40

# Data Analysis Settings
MAX_PREVIEW_ROWS = 100
MAX_FILE_SIZE_MB = 100
