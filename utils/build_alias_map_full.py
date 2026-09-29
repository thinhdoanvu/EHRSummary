#STEP 3. build_alias_map_full.py
#Lưu file build_alias_map_full.py và chạy. Nó tạo alias_map.pkl để load nhanh trong pipeline.

import json, re, unicodedata, pickle
LEX_PATH = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\lexicon_from_mrconso.json"
OUT_PICKLE = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\alias_map_full.pkl"

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

print("Loading", LEX_PATH)
with open(LEX_PATH, encoding="utf-8") as f:
    j = json.load(f)
entries = j.get("entries", j)

alias_map = {}
for entry in entries:
    cui = entry.get("cui")
    canonical = entry.get("canonical")
    for a in entry.get("aliases", []):
        key = normalize(a)
        alias_map.setdefault(key, []).append({
            "cui": cui,
            "canonical": canonical,
            "alias": a,
            "sources": entry.get("sources", [])
        })

print("Aliases:", len(alias_map), "concepts:", len(entries))
with open(OUT_PICKLE, "wb") as fo:
    pickle.dump(alias_map, fo)
print("Saved alias_map to", OUT_PICKLE)
