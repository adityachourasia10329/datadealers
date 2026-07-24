"""
SQLite Database Initialization and Data Access Objects for DataDealers
"""

import sqlite3
import os
import random
from werkzeug.security import generate_password_hash, check_password_hash
from quotes_seed import QUOTES

DB_PATH = os.path.join(os.path.dirname(__file__), "datadealers.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Create Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            downloads_allowed INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create Quotes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            author TEXT NOT NULL
        )
    """)

    # Create Custom Datasets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS custom_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            tag TEXT NOT NULL,
            description TEXT,
            rows INTEGER DEFAULT 0,
            cols INTEGER DEFAULT 0,
            size TEXT,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check quotes count
    cursor.execute("SELECT COUNT(*) FROM quotes")
    count = cursor.fetchone()[0]

    if count == 0:
        cursor.executemany(
            "INSERT INTO quotes (text, author) VALUES (?, ?)",
            [(q["text"], q["author"]) for q in QUOTES]
        )

    conn.commit()
    conn.close()


def create_user(name, email, password, downloads_allowed=1):
    conn = get_db()
    cursor = conn.cursor()
    pwd_hash = generate_password_hash(password)
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, downloads_allowed) VALUES (?, ?, ?, ?)",
            (name, email.strip().lower(), pwd_hash, 1 if downloads_allowed else 0)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {"id": user_id, "name": name, "email": email}
    except sqlite3.IntegrityError:
        conn.close()
        return None


def verify_user(email, password):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    if row and check_password_hash(row["password_hash"], password):
        return {"id": row["id"], "name": row["name"], "email": row["email"]}
    return None


def get_random_quote():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT text, author FROM quotes ORDER BY RANDOM() LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"text": row["text"], "author": row["author"]}
    return {
        "text": "No man ever steps in the same river twice, for it's not the same river and he's not the same man.",
        "author": "Heraclitus"
    }


def add_custom_dataset(name, tag, description, rows, cols, size, file_path):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO custom_datasets (name, tag, description, rows, cols, size, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, tag, description, rows, cols, size, file_path))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        return False


def get_custom_datasets():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, tag, description, rows, cols, size, file_path FROM custom_datasets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Ensure DB is initialized on module import
init_db()

