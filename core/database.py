import sqlite3
import os
from typing import Optional

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_DIR: str = os.path.join(BASE_DIR, "instance")
USERS_DB_PATH: str = os.path.join(INSTANCE_DIR, "users.db")


def get_db_connection() -> sqlite3.Connection:
    """Return a new SQLite connection for the users database."""
    conn: sqlite3.Connection = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_users_db() -> None:
    """Create the users table if it does not already exist."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    conn: sqlite3.Connection = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                signup_date TEXT,
                last_login TEXT
            )
            """
        )
        conn.commit()
        
        # Add columns dynamically if the table already existed without them
        for column_name in ["signup_date", "last_login"]:
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column_name} TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                # Column already exists, safe to skip
                pass
    finally:
        conn.close()
