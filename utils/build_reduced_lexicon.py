# build_reduced_lexicon.py
import sqlite3, json, re
DB = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\umls_simple.sqlite"
MRSTY = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\MRSTY.RRF"
OUT_REDUCED = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\lexicon_reduced.json"

# load MRSTY to get semantic types per CUI
cui_to_stys = {}
with open(MRSTY, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.split('|')
        if len(parts) < 3: continue
        cui = parts[0]; sty = parts[2]
        cui_to_stys.setdefault(cui, set()).add(sty)

# allowed semantic types (tune as needed)
allowed = {"Disease or Syndrome","Sign or Symptom","Injury or Poisoning","Pathologic Function","Neoplastic Process"}

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT cui, str, norm, sab, code, tty, ispref FROM concept_strings")
rows = cur.fetchall()
by_cui = {}
for cui, s, norm, sab, code, tty, ispref in rows:
    stys = cui_to_stys.get(cui, set())
    if not stys.intersection(allowed):
        continue
    rec = by_cui.setdefault(cui, {"cui":cui, "aliases":[], "sources":[],"preferred":None, "stys":list(stys)})
    if s not in rec["aliases"]:
        rec["aliases"].append(s)
    rec["sources"].append({"sab":sab,"code":code,"tty":tty,"ispref":ispref})
    if ispref == 'Y' or tty == 'PT':
        rec["preferred"] = s

out = []
for cui, v in by_cui.items():
    canonical = v["preferred"] or (v["aliases"][0] if v["aliases"] else None)
    out.append({"cui":cui,"canonical":canonical,"aliases":v["aliases"],"sources":v["sources"],"stys":v["stys"]})

with open(OUT_REDUCED,"w",encoding="utf-8") as fo:
    json.dump({"version":"reduced_v1","count":len(out),"entries":out}, fo, ensure_ascii=False, indent=2)
conn.close()
print("Wrote reduced lexicon:", OUT_REDUCED, "entries:", len(out))
