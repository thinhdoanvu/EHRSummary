# build_cui_icd11_map_safe.py
# 3 — Trích n‑grams phổ biến từ 6k posts (để tìm candidate phrases)
# Script extract_top_ngrams.py tạo top_ngrams.txt (unigram/bi/tri) để review.
import sqlite3, os

SRC_DB = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\umls_simple.sqlite"
OUT_DB = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\cui_has_icd11.sqlite"
ICD11_FILE = r"C:\Users\VU\Documents\NLP\datasets\ICD-11-MMS.txt"

# -----------------------------
# Connect databases
# -----------------------------
src_conn = sqlite3.connect(SRC_DB)
src_cur = src_conn.cursor()

out_conn = sqlite3.connect(OUT_DB)
out_cur = out_conn.cursor()

# -----------------------------
# Prepare output table
# -----------------------------
out_cur.executescript("""
CREATE TABLE IF NOT EXISTS cui_icd_map (
  cui  TEXT,
  sab  TEXT,
  code TEXT,
  src  TEXT
);
CREATE INDEX IF NOT EXISTS idx_cui  ON cui_icd_map(cui);
CREATE INDEX IF NOT EXISTS idx_code ON cui_icd_map(code);
""")
out_conn.commit()

BATCH = 10000

def flush(batch):
    out_cur.executemany(
        "INSERT INTO cui_icd_map VALUES (?,?,?,?)",
        batch
    )
    out_conn.commit()
    batch.clear()

# -----------------------------
# 1) Direct ICD-11 in MRCONSO
# -----------------------------
print("Step 1: MRCONSO ICD-11")
src_cur.execute("""
SELECT DISTINCT cui, sab, code
FROM concept_strings
WHERE upper(sab) LIKE '%ICD11%'
""")

batch = []
for cui, sab, code in src_cur:
    batch.append((cui, sab, code, "MRCONSO_ICD11"))
    if len(batch) >= BATCH:
        flush(batch)
flush(batch)

# -----------------------------
# 2) Other ICD (ICD10CM, etc.)
# -----------------------------
print("Step 2: MRCONSO ICD-10 / ICD*")
src_cur.execute("""
SELECT DISTINCT cui, sab, code
FROM concept_strings
WHERE upper(sab) LIKE '%ICD10%'
   OR upper(sab) LIKE 'ICD%'
""")

batch = []
for cui, sab, code in src_cur:
    batch.append((cui, sab, code, "MRCONSO_ICD"))
    if len(batch) >= BATCH:
        flush(batch)
flush(batch)

# -----------------------------
# 3) ICD-11 MMS file mapping (optional)
# -----------------------------
if ICD11_FILE and os.path.exists(ICD11_FILE):
    print("Step 3: ICD-11 MMS file")
    codes = set()
    with open(ICD11_FILE, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                codes.add(line.split()[0])

    for code in codes:
        src_cur.execute(
            "SELECT DISTINCT cui, sab FROM concept_strings WHERE code = ?",
            (code,)
        )
        rows = [(cui, sab, code, "ICD11_FILE") for cui, sab in src_cur.fetchall()]
        if rows:
            out_cur.executemany(
                "INSERT INTO cui_icd_map VALUES (?,?,?,?)",
                rows
            )
            out_conn.commit()

# -----------------------------
# Done
# -----------------------------
src_conn.close()
out_conn.close()
print("DONE. Safe incremental write to", OUT_DB)
