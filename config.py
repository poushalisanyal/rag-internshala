import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================================
# API Configuration
# ==========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini Model
LLM_MODEL = "gemini-2.5-flash"

# ==========================================================
# Embedding Configuration
# ==========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384

# ==========================================================
# Document Chunking
# ==========================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ==========================================================
# Retrieval Configuration
# ==========================================================

TOP_K = 4

# ==========================================================
# Vector Database
# ==========================================================

VECTOR_DB_PATH = "vector_db"

# ==========================================================
# Data Folder
# ==========================================================

DATA_FOLDER = "data"

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"

# ==========================================================
# Evaluation
# ==========================================================

EVAL_FILE = "evaluation_questions.json"
RESULTS_FILE = "evaluation_results.json"

# ==========================================================
# Cost Analysis (Used in evaluator.py)
# ==========================================================

MANAGED_DB_COST_PER_MILLION = 120.0  # Example ($/month)
FAISS_COST_PER_MILLION = 0.0          # Local storage

# ==========================================================
# Application
# ==========================================================

HOST = "0.0.0.0"
PORT = 8000
DEBUG = True