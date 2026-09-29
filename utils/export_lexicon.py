# STEP 2. export_lexicon.py
import sqlite3, json
DB = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\umls_simple.sqlite"
OUT = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\lexicon_from_mrconso.json"

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT cui, str, norm, sab, code, tty, ispref FROM concept_strings")
rows = cur.fetchall()
by_cui = {}
for cui, s, norm, sab, code, tty, ispref in rows:
    rec = by_cui.setdefault(cui, {"cui":cui, "aliases":[], "sources":[],"preferred":None})
    if s not in rec["aliases"]:
        rec["aliases"].append(s)
    rec["sources"].append({"sab":sab,"code":code,"tty":tty,"ispref":ispref})
    if ispref == 'Y' or tty == 'PT':
        rec["preferred"] = s
out = []
for cui, v in by_cui.items():
    canonical = v["preferred"] or (v["aliases"][0] if v["aliases"] else None)
    out.append({"cui":cui,"canonical":canonical,"aliases":v["aliases"],"sources":v["sources"]})
with open(OUT,"w",encoding="utf-8") as fo:
    json.dump({"version":"v1","count":len(out),"entries":out}, fo, ensure_ascii=False, indent=2)
conn.close()
print("Wrote", len(out), "concepts to", OUT)
