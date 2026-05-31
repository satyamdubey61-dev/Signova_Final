import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.database import init_users_db, get_db_connection
from services.auth_service import AuthService

# Initialize database
init_users_db()

# Try registering a test user
email = "test@example.com"
password = "testpassword"
name = "Test User"

# Clean up existing test user if any
conn = get_db_connection()
conn.execute("DELETE FROM users WHERE email = ?", (email,))
conn.commit()
conn.close()

# 1. Test creation
success, msg = AuthService.create_user(name, email, password)
print(f"Signup result: success={success}, msg={msg}")

# 2. Test authentication
auth_success, user_dict = AuthService.authenticate_user(email, password)
print(f"Auth result: success={auth_success}, user={user_dict}")

if auth_success and user_dict:
    print("User authenticated successfully! DB operations are completely correct!")
else:
    print("User authentication failed!")
