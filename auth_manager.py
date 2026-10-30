"""
Authentication and Role-Based Access Control Module
Provides SQLite-backed user management with salted SHA-256 password hashing.
Roles: 'admin' (Placement Cell / Faculty) and 'student'.
"""

import sqlite3
import hashlib
import os
import secrets
from typing import Optional, Dict, List

DB_PATH = os.path.join(os.path.dirname(__file__), "placement_users.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hashes password with SHA-256 and unique salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    salted = f"{salt}{password}".encode("utf-8")
    pwd_hash = hashlib.sha256(salted).hexdigest()
    return pwd_hash, salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    pwd_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(pwd_hash, expected_hash)


def init_auth_db():
    """Initializes tables and seeds default administrative & student credentials."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'student')),
            full_name TEXT NOT NULL,
            email TEXT,
            student_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed default accounts if database is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]

    if count == 0:
        default_users = [
            ("admin", "admin123", "admin", "Prof. K. Sharma (Placement Head)", "placement@college.edu", None),
            ("coordinator", "coord123", "admin", "Dr. A. Verma (TPO Coordinator)", "tpo@college.edu", None),
            ("pranay", "student123", "student", "Pranay Kumar", "pranay@student.edu", "STU202401"),
            ("rahul", "student123", "student", "Rahul Gupta", "rahul@student.edu", "STU202402"),
            ("sneha", "student123", "student", "Sneha Patil", "sneha@student.edu", "STU202403"),
            ("ananya", "student123", "student", "Ananya Reddy", "ananya@student.edu", "STU202404"),
        ]

        for username, pwd, role, name, email, stu_id in default_users:
            pwd_hash, salt = hash_password(pwd)
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, role, full_name, email, student_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (username, pwd_hash, salt, role, name, email, stu_id))

        conn.commit()
    conn.close()


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticates user and returns dict with user profile if valid."""
    init_auth_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    if verify_password(password, row["salt"], row["password_hash"]):
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "full_name": row["full_name"],
            "email": row["email"],
            "student_id": row["student_id"],
        }
    return None


def register_user(username: str, password: str, role: str, full_name: str, email: str = "", student_id: str = "") -> tuple[bool, str]:
    """Registers a new user into the database."""
    init_auth_db()
    if not username or not password or not full_name:
        return False, "Username, password, and full name are required."

    if role not in ("admin", "student"):
        return False, "Role must be either 'admin' or 'student'."

    pwd_hash, salt = hash_password(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, salt, role, full_name, email, student_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username.strip(), pwd_hash, salt, role, full_name.strip(), email.strip(), student_id.strip() if student_id else None))
        conn.commit()
        conn.close()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"Username '{username}' already exists."
    except Exception as e:
        conn.close()
        return False, f"Registration failed: {str(e)}"


def get_all_users() -> List[Dict]:
    """Returns list of all registered users (for admin user management)."""
    init_auth_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, full_name, email, student_id, created_at FROM users ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user(user_id: int) -> bool:
    """Deletes a user account."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    init_auth_db()
    print("Database initialized successfully.")
    admin_test = authenticate_user("admin", "admin123")
    print("Admin Auth Test:", admin_test)
