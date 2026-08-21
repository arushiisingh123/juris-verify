import sqlite3
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "matrix.db")

def init_db(db_path=DB_PATH):
    """Initialize the contradiction matrix database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contradiction_matrix (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            claim_id TEXT,
            source_doc_id TEXT,
            label TEXT,
            confidence REAL,
            evidence_span TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_record(run_id, claim_id, source_doc_id, label, confidence, evidence_span, db_path=DB_PATH):
    """Insert a verification record into the contradiction matrix."""
    # Ensure DB exists
    init_db(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO contradiction_matrix (run_id, claim_id, source_doc_id, label, confidence, evidence_span)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (run_id, claim_id, source_doc_id, label, confidence, evidence_span))
    conn.commit()
    conn.close()

def get_records_for_claim(claim_id, db_path=DB_PATH):
    """Retrieve all verification records for a specific claim."""
    if not os.path.exists(db_path):
        return []
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, run_id, claim_id, source_doc_id, label, confidence, evidence_span, timestamp
        FROM contradiction_matrix
        WHERE claim_id = ?
        ORDER BY timestamp ASC
    ''', (claim_id,))
    
    records = []
    for row in cursor.fetchall():
        records.append({
            "id": row[0],
            "run_id": row[1],
            "claim_id": row[2],
            "source_doc_id": row[3],
            "label": row[4],
            "confidence": row[5],
            "evidence_span": row[6],
            "timestamp": row[7]
        })
    conn.close()
    return records
