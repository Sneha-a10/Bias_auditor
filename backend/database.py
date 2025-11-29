"""
Database setup and models for the Bias Auditor.
Uses SQLite for simplicity in MVP.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "auditor.db"


def get_connection():
    """Get a database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            error_message TEXT
        )
    """)
    
    conn.commit()
    conn.close()


class RunStatus:
    """Run status constants."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class RunDB:
    """Database operations for runs."""
    
    @staticmethod
    def create(run_id: str, config: dict) -> None:
        """Create a new run."""
        conn = get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        cursor.execute(
            """
            INSERT INTO runs (id, status, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, RunStatus.PENDING, json.dumps(config), now, now)
        )
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def get(run_id: str) -> Optional[dict]:
        """Get a run by ID."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row["id"],
                "status": row["status"],
                "config": json.loads(row["config_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "error_message": row["error_message"]
            }
        return None
    
    @staticmethod
    def list_all() -> list[dict]:
        """List all runs."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM runs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row["id"],
                "status": row["status"],
                "config": json.loads(row["config_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "error_message": row["error_message"]
            }
            for row in rows
        ]
    
    @staticmethod
    def update_status(run_id: str, status: str, error_message: Optional[str] = None) -> None:
        """Update run status."""
        conn = get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        cursor.execute(
            """
            UPDATE runs
            SET status = ?, updated_at = ?, error_message = ?
            WHERE id = ?
            """,
            (status, now, error_message, run_id)
        )
        
        conn.commit()
        conn.close()


# Initialize database on module import
init_db()
