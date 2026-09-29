# 1 — Script: build_cui_sty_map.py (MRSTY → CUI → semantic types)
# Lưu file build_cui_sty_map.py và chạy. Kết quả: cui_to_stys.json.
# build_cui_sty_map.py
import json, sys
mrsty = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\MRSTY.RRF"
out = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\cui_to_stys.json"

cui_to_stys = {}
with open(mrsty, encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        cui = parts[0].strip()
        sty = parts[1].strip()
        if cui:
            cui_to_stys.setdefault(cui, set()).add(sty)
# convert sets to lists
cui_to_stys = {k: list(v) for k,v in cui_to_stys.items()}
with open(out, 'w', encoding='utf-8') as fo:
    json.dump(cui_to_stys, fo, ensure_ascii=False, indent=2)
print("Wrote", len(cui_to_stys), "CUIs to", out)
