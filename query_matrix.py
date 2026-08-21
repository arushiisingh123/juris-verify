import hashlib
from src.matrix import get_records_for_claim

claims = [
    "Test claim - contradicted",
    "Test claim - partially supported",
    "Test claim - exhausted",
    "1 in 5 million in UK have abnormal PrP positivity."
]

for claim in claims:
    claim_id = hashlib.sha256(claim.encode('utf-8')).hexdigest()
    print(f"\n--- Records for claim: '{claim}' (ID: {claim_id[:8]}...) ---")
    records = get_records_for_claim(claim_id)
    if not records:
        print("  No records found.")
    for r in records:
        print(f"  Run ID: {r['run_id']} | Label: {r['label']:<15} | Conf: {r['confidence']:.4f} | TS: {r['timestamp']}")
