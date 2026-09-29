# 3 — Trích n‑grams phổ biến từ 6k posts (để tìm candidate phrases)
# Script extract_top_ngrams.py tạo top_ngrams.txt (unigram/bi/tri) để review.
# extract_top_ngrams.py
import csv, re, unicodedata, glob, os
from collections import Counter

CSV_DIR = r"C:\Users\VU\Documents\NLP\Demo2\datasets\individual_posts"
OUT = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\top_ngrams.txt"

def normalize(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

cnt = Counter()

csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

for csv_path in csv_files:
    print("Processing:", os.path.basename(csv_path))
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get("post_title","") + " " + r.get("post_content","")).strip()
            tokens = normalize(text).split()
            for n in (1, 2, 3):
                cnt.update(ngrams(tokens, n))

with open(OUT, "w", encoding="utf-8") as fo:
    for k, v in cnt.most_common(20000):
        fo.write(f"{k}\t{v}\n")

print("Wrote top ngrams to", OUT)
