#STEP 1.  umls_to_sqlite.py

import os, re, sqlite3

MRCONSO = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\MRCONSO.RRF"
OUT_DB = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\umls_simple.sqlite"
LANGS = {"ENG"}   # chỉ lấy tiếng Anh; đổi nếu cần
SABS_KEEP = None  # hoặc set e.g. {"SNOMEDCT_US","ICD11"}

def normalize(s):
    if not s: return ""
    s = s.strip().lower()
    s = re.sub(r'\s+',' ', s)
    s = re.sub(r'[^\w\s\-]', ' ', s)
    return s

def create_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE concept_strings (
      id INTEGER PRIMARY KEY,
      cui TEXT,
      sab TEXT,
      code TEXT,
      tty TEXT,
      ispref TEXT,
      str TEXT,
      norm TEXT
    );
    """)
    cur.execute("CREATE INDEX idx_cui ON concept_strings(cui);")
    conn.commit()
    return conn

def import_mrconso(conn, path):
    cur = conn.cursor()
    count = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) < 15: continue
            cui = parts[0]; lat = parts[1]; sab = parts[11]; tty = parts[12]; code = parts[13]; string = parts[14].strip()
            if LANGS and lat.upper() not in LANGS: continue
            if SABS_KEEP and sab not in SABS_KEEP: continue
            norm = normalize(string)
            cur.execute("INSERT INTO concept_strings (cui,sab,code,tty,ispref,str,norm) VALUES (?,?,?,?,?,?,?)",
                        (cui,sab,code,tty,parts[6],string,norm))
            count += 1
            if count % 50000 == 0:
                conn.commit()
                print("Inserted", count)
    conn.commit()
    print("Total inserted:", count)

if __name__ == "__main__":
    if not os.path.exists(MRCONSO):
        print("MRCONSO.RRF not found")
        raise SystemExit(1)
    conn = create_db(OUT_DB)
    import_mrconso(conn, MRCONSO)
    conn.close()
    print("Done. DB:", OUT_DB)
