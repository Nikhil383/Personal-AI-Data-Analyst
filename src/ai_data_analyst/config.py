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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# Whether a valid Gemini API key is configured. The app starts without one,
# but LLM-backed nodes (text-to-SQL, natural language answers) require it.
HAS_API_KEY = bool(GOOGLE_API_KEY) and GOOGLE_API_KEY != "your_api_key_here"

# LLM Configuration
GEMINI_MODEL = "gemini-2.5-flash"

# LLM Parameters
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.2
TOP_P = 0.95
TOP_K = 40

# LangGraph / SQL configuration
MAX_SQL_RETRIES = 2

# Data Analysis Settings
MAX_PREVIEW_ROWS = 100
MAX_FILE_SIZE_MB = 100

# Default message returned when a question cannot be answered from the data
NO_INFO_MESSAGE = "No info available"
