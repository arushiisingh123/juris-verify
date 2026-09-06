import sqlite3
import hashlib

conn = sqlite3.connect("matrix.db")
cursor = conn.cursor()

claim = "1 in 5 million in UK have abnormal PrP positivity."
claim_id = hashlib.sha256(claim.encode('utf-8')).hexdigest()

cursor.execute('''
    SELECT id, run_id, claim_id, source_doc_id, label, confidence, evidence_span, timestamp
    FROM contradiction_matrix
    WHERE claim_id = ?
    ORDER BY id ASC
''', (claim_id,))

rows = cursor.fetchall()
print(f"Total rows in DB for claim: {len(rows)}")
for r in rows:
    print(f"ID: {r[0]} | Run: {r[1]} | Doc: {r[3]} | Label: {r[4]} | Conf: {r[5]:.4f} | TS: {r[7]}")
    print(f"   Evidence: {r[6][:100]}...\n")

conn.close()
