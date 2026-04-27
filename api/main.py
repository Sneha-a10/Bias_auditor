"""
Vercel entrypoint - re-exports the FastAPI app from backend/main.py
with proper sys.path setup for sibling module imports.
"""
import sys
import os

# Add backend/ to the path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app  # noqa: F401, E402
