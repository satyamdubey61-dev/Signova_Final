from core.database import get_db_connection
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from utils.logger import logger

class AuthService:
    @staticmethod
    def create_user(name, email, password):
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)),
            )
            conn.commit()
            logger.info(f"New user created: {email}")
            return True, "Account created successfully."
        except sqlite3.IntegrityError:
            logger.warning(f"Signup failed. Account with email already exists: {email}")
            return False, "An account with this email already exists."
        except Exception as e:
            logger.error(f"Error during signup for {email}: {e}")
            return False, "An error occurred during signup."
        finally:
            conn.close()

    @staticmethod
    def authenticate_user(email, password):
        conn = get_db_connection()
        try:
            user = conn.execute(
                "SELECT id, name, email, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        except Exception as e:
            logger.error(f"Error during login for {email}: {e}")
            return False, None
        finally:
            conn.close()

        if user is None or not check_password_hash(user["password_hash"], password):
            logger.warning(f"Failed login attempt for: {email}")
            return False, None
            
        logger.info(f"User logged in: {email}")
        return True, {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
        }
