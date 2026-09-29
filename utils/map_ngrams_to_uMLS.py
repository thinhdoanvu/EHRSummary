# map_ngrams_to_umls_safe.py
# 4 — Map top n‑grams → alias_map (exact → fuzzy) để thu synonyms/slang
# Script map_ngrams_to_uMLS.py sẽ đọc top_ngrams.txt, map vào alias_map_full.pkl (exact then fuzzy), và xuất mapped_candidates.json (các n‑gram có match UMLS CUI).
import json, pickle, unicodedata, re, os
from rapidfuzz import process, fuzz

ALIAS_PKL = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\alias_map_full.pkl"
NGRAMS = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\top_ngrams.txt"
OUT = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\mapped_candidates.jsonl"
FUZZY_THRESHOLD = 80

def normalize(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------------
# Load alias map
# -----------------------------
with open(ALIAS_PKL, "rb") as f:
    alias_map = pickle.load(f)

keys = list(alias_map.keys())

# -----------------------------
# Resume support
# -----------------------------
seen = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                seen.add(obj["phrase"])
            except Exception:
                pass
print("Resume from", len(seen), "phrases")

# -----------------------------
# Mapping
# -----------------------------
with open(OUT, "a", encoding="utf-8") as fo:
    with open(NGRAMS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            phrase = line.split("\t", 1)[0].strip()
            if phrase in seen:
                continue

            pn = normalize(phrase)

            # exact
            if pn in alias_map:
                out = {
                    "phrase": phrase,
                    "method": "exact",
                    "matches": alias_map[pn][:5]
                }
                fo.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            # fuzzy
            matches = process.extract(
                pn, keys,
                scorer=fuzz.token_set_ratio,
                limit=5
            )

            good = []
            best = 0
            for key, score, _ in matches:
                if score >= FUZZY_THRESHOLD:
                    best = max(best, score)
                    good.extend(alias_map.get(key, [])[:3])

            if good:
                out = {
                    "phrase": phrase,
                    "method": "fuzzy",
                    "score": best,
                    "matches": good
                }
                fo.write(json.dumps(out, ensure_ascii=False) + "\n")

            if i % 1000 == 0:
                fo.flush()

print("DONE. Results written incrementally to", OUT)
