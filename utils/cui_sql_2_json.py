import sqlite3, json

DB = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\cui_has_icd11.sqlite"
OUT = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\cui_has_icd11.json"

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("""
SELECT cui, sab, code
FROM cui_icd_map
""")

cui_map = {}
for cui, sab, code in cur:
    cui_map.setdefault(cui, []).append({
        "sab": sab,
        "code": code
    })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(cui_map, f, ensure_ascii=False, indent=2)

conn.close()
print("Exported JSON to", OUT)
