from typing import Any, Dict, List, Optional, Tuple, Union

from core.database import get_db_connection, USERS_DB_PATH
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import json
from utils.logger import logger

USERS_JSON_PATH: str = os.path.join(os.path.dirname(USERS_DB_PATH), "users.json")


def _fallback_create_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    try:
        os.makedirs(os.path.dirname(USERS_JSON_PATH), exist_ok=True)
        users: List[Dict[str, Any]] = []
        if os.path.exists(USERS_JSON_PATH):
            with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
                try:
                    users = json.load(f)
                except Exception:
                    users = []

        # Check if email exists
        for u in users:
            if u.get("email") == email:
                logger.warning(f"Fallback signup failed. Account already exists: {email}")
                return False, "An account with this email already exists."

        new_user: Dict[str, Any] = {
            "id": len(users) + 1,
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password)
        }
        users.append(new_user)
        with open(USERS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=4)
        logger.info(f"Fallback: New user created in JSON storage: {email}")
        return True, "Account created successfully (Fallback)."
    except Exception as e:
        logger.error(f"Fallback signup failed for {email}: {e}")
        return False, "An error occurred during signup."


def _fallback_authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        if not os.path.exists(USERS_JSON_PATH):
            return False, None
        with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                users: List[Dict[str, Any]] = json.load(f)
            except Exception:
                return False, None

        for u in users:
            if u.get("email") == email:
                pw_hash: str = str(u.get("password_hash", ""))
                if check_password_hash(pw_hash, password):
                    logger.info(f"Fallback: User logged in from JSON storage: {email}")
                    return True, {
                        "id": u.get("id"),
                        "name": u.get("name"),
                        "email": u.get("email")
                    }
                break
        logger.warning(f"Fallback: Failed login attempt for: {email}")
        return False, None
    except Exception as e:
        logger.error(f"Fallback login failed for {email}: {e}")
        return False, None


class AuthService:
    @staticmethod
    def create_user(name: str, email: str, password: str) -> Tuple[bool, str]:
        try:
            conn: sqlite3.Connection = get_db_connection()
        except Exception as db_err:
            logger.warning(f"SQLite connection failed: {db_err}. Falling back to JSON storage.")
            return _fallback_create_user(name, email, password)

        try:
            conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)),
            )
            conn.commit()
            logger.info(f"New user created in SQLite: {email}")
            return True, "Account created successfully."
        except sqlite3.IntegrityError:
            logger.warning(f"SQLite signup failed. Account with email already exists: {email}")
            return False, "An account with this email already exists."
        except Exception as e:
            logger.error(f"SQLite signup error for {email}: {e}. Trying JSON fallback...")
            return _fallback_create_user(name, email, password)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    @staticmethod
    def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        try:
            conn: sqlite3.Connection = get_db_connection()
        except Exception as db_err:
            logger.warning(f"SQLite connection failed: {db_err}. Falling back to JSON storage.")
            return _fallback_authenticate_user(email, password)

        try:
            row: Optional[sqlite3.Row] = conn.execute(
                "SELECT id, name, email, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        except Exception as e:
            logger.error(f"SQLite login query failed for {email}: {e}. Trying JSON fallback...")
            return _fallback_authenticate_user(email, password)
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if row is None:
            # Check JSON storage fallback
            logger.info(f"User not found in SQLite: {email}. Checking JSON fallback...")
            return _fallback_authenticate_user(email, password)

        pw_hash: str = str(row["password_hash"])
        if not check_password_hash(pw_hash, password):
            logger.warning(f"Failed login attempt for: {email}")
            return False, None

        logger.info(f"User logged in: {email}")
        return True, {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
        }
