import sqlite3
from datetime import datetime

DB_PATH = "bmi_history.db"

# =========================
# Init DB
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS bmi_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            confidence REAL NOT NULL,
            has_face INTEGER NOT NULL,
            face_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# =========================
# Save history
# =========================
def save_bmi_history(class_id, confidence, has_face, face_count):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO bmi_history
        (class_id, confidence, has_face, face_count, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        class_id,
        confidence,
        int(has_face),
        face_count,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

# =========================
# Get history
# =========================
def get_bmi_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM bmi_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]
