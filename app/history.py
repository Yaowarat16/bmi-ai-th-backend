import sqlite3
from datetime import datetime
import os

# =========================
# Database Config
# =========================
# เก็บไฟล์ DB ไว้ที่ root ของโปรเจกต์
DB_PATH = os.getenv("BMI_HISTORY_DB", "bmi_history.db")


# =========================
# Init Database
# =========================
def init_db():
    """
    สร้างตารางเก็บประวัติ BMI
    เรียกใช้ครั้งเดียวตอน API start
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS bmi_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bmi_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        has_face INTEGER NOT NULL,
        face_count INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# =========================
# Save History
# =========================
def save_bmi_history(bmi_class: str, confidence: float, has_face: bool, face_count: int):
    """
    บันทึกประวัติผล BMI
    - ไม่เปรียบเทียบ
    - ไม่ผูก user
    """

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO bmi_history
        (bmi_class, confidence, has_face, face_count, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            bmi_class,
            float(confidence),
            int(has_face),
            int(face_count),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()
    conn.close()


# =========================
# Optional: Get History (debug / future use)
# =========================
def get_all_history(limit: int = 100):
    """
    ดึงประวัติล่าสุด (เผื่อใช้ debug หรือขยายระบบในอนาคต)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, bmi_class, confidence, has_face, face_count, created_at
        FROM bmi_history
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cur.fetchall()
    conn.close()
    return rows
