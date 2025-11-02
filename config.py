
"""
Configuration file for the RAG pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- File Paths ---
PDF_PATH = "data/jemh109.pdf"
#QDRANT_PATH = "./qdrant_db" # Directory to store local Qdrant database
QDRANT_URL = "http://localhost:6333"
# --- Qdrant Settings ---
COLLECTION_NAME = "jmeh_multimodal"

# --- Model Settings (Ollama) ---
# Make sure you have pulled these models:
# $ ollama pull nomic-embed-text
# $ ollama pull llava
# $ ollama pull llama3
TEXT_EMBEDDING_MODEL = "nomic-embed-text"
IMAGE_SUMMARY_MODEL = "llava-cpu" # Local VLM for summarizing images
LLM_MODEL = "llama3" # For RAG answers
OLLAMA_BASE_URL = "http://localhost:11434"

# --- HuggingFace (for unstructured) ---
# Used by unstructured to download helper models for layout detection
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
