import sqlite3
from datetime import datetime
import pytz

# =========================
# Database Config
# =========================
DB_PATH = "bmi_history.db"

# =========================
# Init Database
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS bmi_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            bmi_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            has_face INTEGER NOT NULL,
            face_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

# =========================
# Save BMI History
# =========================
def save_bmi_history(
    class_id: int,
    bmi_label: str,
    confidence: float,
    has_face: bool,
    face_count: int
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ✅ เวลาไทย (UTC+7)
    tz = pytz.timezone("Asia/Bangkok")
    created_at = datetime.now(tz).strftime("%d/%m/%Y %H:%M:%S")

    c.execute("""
        INSERT INTO bmi_history
        (class_id, bmi_label, confidence, has_face, face_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        class_id,
        bmi_label,
        confidence,
        int(has_face),
        face_count,
        created_at
    ))

    conn.commit()
    conn.close()

# =========================
# Get BMI History
# =========================
def get_bmi_history(limit: int = 10):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT
            id,
            class_id,
            bmi_label,
            confidence,
            has_face,
            face_count,
            created_at
        FROM bmi_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = c.fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "class_id": r["class_id"],
            "bmi_label": r["bmi_label"],
            "confidence": r["confidence"],
            "has_face": bool(r["has_face"]),
            "face_count": r["face_count"],
            "created_at": r["created_at"]
        }
        for r in rows
    ]
