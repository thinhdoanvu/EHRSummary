#!/usr/bin/env python3
# DiseaseCouncil.py — Fixed version
# Fixes applied:
#   1. First-person disease filter (không extract bệnh từ tên vaccine, thú cưng, hypothetical)
#   2. JSON serialization bug (disease field chứa raw dict)
#   3. ICD blacklist cho severity qualifiers
#   4. ICD correction map cho các lỗi phổ biến

import os
import csv
import unicodedata
import pickle
from collections import Counter
import spacy
from rapidfuzz import process, fuzz
from openai import OpenAI
import json
import re

# ---------------------------
# CONFIG
# ---------------------------
CSV_IN      = r"C:\Users\VU\Documents\NLP\Demo2\datasets\individual_posts\1ac3d1t.csv"
OUT_JSONL   = r"C:\Users\VU\Documents\NLP\Demo2\datasets\raw_reports_txt\results_diseases_only.jsonl"

ALIAS_PKL       = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\alias_map_full.pkl"
CUI_TO_STYS     = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\cui_to_stys.json"
MAPPED_NGRAMS   = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\mapped_candidates.json"
VACCINE_JSON    = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\vaccine_to_diseases.json"
FUZZY_THRESHOLD = 95
TOP_K           = 5

# =========================================================
# MEDICAL KEYWORDS & SEMANTIC FILTERS
# =========================================================

DISEASE_SUFFIXES = (
    "itis", "osis", "oma", "pathy", "algia",
    "emia", "penia", "rhea", "rrhage", "lysis",
    "ectasis", "malacia", "uria", "clasis", "stasis",
    "megaly", "stenosis", "sclerosis", "blastoma",
    "plasty", "pathies", "cele", "spasm", "necrosis",
    "fibrosis", "dysplasia", "atrophy", "hypertrophy",
    "hyperplasia", "ischemia", "infarction", "embolism",
    "thrombosis", "sepsis", "shock", "toxicity", "failure",
    "insufficiency", "collapse"
)

SYMPTOM_KEYWORDS = {
    "cough", "dyspnea", "tachypnea", "apnea",
    "bronchitis", "pneumonia", "pharyngitis",
    "wheezing", "hypoxia", "respiratory distress",
    "diarrhea", "constipation", "abdominal pain",
    "gastritis", "colitis", "nausea", "vomiting",
    "renal failure", "nephritis", "hematuria",
    "proteinuria", "kidney injury",
    "diabetes", "thyroiditis", "hypothyroidism",
    "hyperthyroidism", "obesity", "hormonal imbalance",
    "anemia", "thrombocytopenia", "leukemia", "lymphoma",
    "blood clot", "thrombosis", "DVT",
    "pulmonary embolism", "autoimmune", "inflammation",
    "depression", "anxiety", "OCD", "psychosis",
    "stroke", "dementia", "seizure",
    "brain fog", "confusion", "memory loss",
    "tremor", "ataxia", "vertigo", "paralysis",
    "numbness", "tingling", "neuropathy",
    "myocarditis", "pericarditis", "arrhythmia",
    "tachycardia", "bradycardia", "palpitation",
    "chest pain", "heart failure", "hypertension",
    "hypotension", "cardiac arrest",
    "rash", "itchy", "swelling", "redness",
    "abscess", "cellulitis", "blister",
    "hives", "edema", "nodule",
    "fever", "chills", "fatigue", "malaise",
    "headache", "dizziness", "syncope",
    "anaphylaxis", "lethargy",
    "pregnant", "pregnancy", "gestational",
    "menstrual", "amenorrhea", "irregular period",
    "infertility", "miscarriage", "postpartum",
    "CRP", "ESR", "d-dimer", "troponin",
    "PCR", "CT scan", "MRI", "x-ray", "ultrasound",
    "infection", "reaction", "allergy", "injury",
    "tumor", "cancer", "syndrome", "disorder",
    "disease", "toxic", "poisoning",
    "mutation", "alteration"
}

ALLOWED_STYS = {
    "Disease or Syndrome",
    "Sign or Symptom",
    "Pathologic Function",
    "Injury or Poisoning",
    "Neoplastic Process",
    "Congenital Abnormality",
    "Finding",
    "Laboratory Procedure",
    "Laboratory or Test Result",
    "Diagnostic Procedure",
    "Clinical Attribute",
    "Anatomical Abnormality",
    "Body Location or Region",
    "Organism Function",
    "Mental or Behavioral Dysfunction",
    "Cell or Molecular Dysfunction",
    "Genetic Function",
    "Gene or Genome",
    "Therapeutic or Preventive Procedure",
    "Pharmacologic Substance",
    "Medical Device",
    "Health Care Activity",
    "Individual Behavior",
    "Social Behavior"
}

# FIX: ICD codes không hợp lệ khi dùng standalone
ICD_BLACKLIST_STANDALONE = {
    "XS5D",   # Mild pain — severity qualifier
    "XS5W",   # Mild severity qualifier
    "XS2R",   # Moderate severity qualifier
    "XS0T",   # Severe severity qualifier
    "XS1M",   # Profound severity qualifier
    "XE82Y",  # Welding light exposure — không liên quan vaccine side effects
}

# FIX: Mapping các ICD errors phổ biến
ICD_CORRECTION_MAP = {
    "MF30": "LD90.Z",    # Breast lump → Injection site reaction
    "MB24.4": "MG22",    # Apathy (psychiatric) → Fatigue
    "8A80": "MG30.0",    # Migraine → Headache unspecified
}

# ---------------------------
# LOAD MODELS / RESOURCES
# ---------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    raise SystemExit("spaCy model en_core_web_sm missing. Run: python -m spacy download en_core_web_sm")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_noise(p: str) -> bool:
    p = (p or "").strip()
    if len(p) <= 2:
        return True
    if re.fullmatch(r'[ivxlcdmIVXLCDM]', p):
        return True
    if re.fullmatch(r'[A-Za-z]', p):
        return True
    return False


# alias map
if not os.path.exists(ALIAS_PKL):
    raise SystemExit(f"alias_map file not found: {ALIAS_PKL}")
with open(ALIAS_PKL, "rb") as f:
    alias_map = pickle.load(f)
alias_keys = list(alias_map.keys())

# semantic types
if not os.path.exists(CUI_TO_STYS):
    raise SystemExit(f"cui_to_stys file not found: {CUI_TO_STYS}")
with open(CUI_TO_STYS, encoding='utf-8') as f:
    cui_to_stys = json.load(f)

# optional mapped ngrams
mapped_ngrams = {}
if MAPPED_NGRAMS and os.path.exists(MAPPED_NGRAMS):
    with open(MAPPED_NGRAMS, encoding='utf-8') as f:
        mapped_ngrams = json.load(f)


def has_disease_sty(cui: str) -> bool:
    if not cui:
        return False
    stys = cui_to_stys.get(cui, [])
    return bool(set(stys) & ALLOWED_STYS)


# ---------------------------
# LOAD VACCINE → DISEASE MAP
# ---------------------------
if not os.path.exists(VACCINE_JSON):
    raise SystemExit(f"vaccine_to_diseases.json not found: {VACCINE_JSON}")
with open(VACCINE_JSON, encoding="utf-8") as f:
    VACCINE_TO_DISEASES = json.load(f)

# ---------------------------
# MAPPING / MATCHING
# ---------------------------
def map_to_aliases(norm_phrase: str):
    if not norm_phrase:
        return []
    if norm_phrase in alias_map:
        return alias_map[norm_phrase]
    matches = process.extract(norm_phrase, alias_keys, scorer=fuzz.token_set_ratio, limit=20)
    candidates = []
    for key, score, _ in matches:
        if score < FUZZY_THRESHOLD:
            continue
        for e in alias_map.get(key, []):
            cui = e.get("cui")
            canonical = (e.get("canonical") or e.get("alias") or "") or ""
            candidates.append((bool(cui and has_disease_sty(cui)), len(canonical), int(score), e))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [c[-1] for c in candidates]


def looks_like_disease_by_text(text: str) -> bool:
    c = normalize(text or "")
    if not c:
        return False
    for suf in DISEASE_SUFFIXES:
        if c.endswith(suf):
            return True
    for kw in SYMPTOM_KEYWORDS:
        if kw in c:
            return True
    return False


# ---------------------------
# EXTRACTION
# ---------------------------
def candidate_phrases(text: str):
    doc = nlp(text)
    phrases = []
    for ent in doc.ents:
        phrases.append(ent.text)
    for nc in doc.noun_chunks:
        phrases.append(nc.text)
    s_norm = normalize(text)
    for phrase in mapped_ngrams.keys():
        if phrase and phrase in s_norm:
            phrases.append(phrase)
    counts = Counter([normalize(p) for p in phrases if p.strip()])
    return counts, phrases


def expand_vaccine_terms(text: str):
    """Tìm các từ vaccine trong text và expand thành danh sách bệnh."""
    diseases = []
    norm_text = normalize(text)
    for vaccine, dis_list in VACCINE_TO_DISEASES.items():
        if vaccine in norm_text:
            diseases.extend(dis_list)
    return diseases


def clean_disease_term(term) -> str:
    """
    FIX 1: Làm sạch tên bệnh.
    Xử lý trường hợp term là dict thay vì string (JSON serialization bug).
    """
    # FIX: Handle dict input
    if isinstance(term, dict):
        term = str(term.get("disease") or term.get("name") or term.get("term") or "")

    if not isinstance(term, str):
        term = str(term)

    t = term.lower().strip()

    # Loại bỏ JSON artifacts
    t_clean = t.strip("{}").strip("'\"").strip()
    if t_clean != t:
        t = t_clean

    if t in ["disorder", "bleeding", "disease", "syndrome"]:
        return ""
    if any(w in t for w in ["history", "descriptor", "question", "score", "value", "record", "doctor"]):
        return ""
    # giữ bệnh ngắn gọn (≤ 4 từ)
    if len(t.split()) > 4:
        return ""
    return term.strip()


def infer_for_row(post_id: str, title: str, body: str):
    """
    Trích xuất danh sách bệnh từ một post (title + body).
    Ưu tiên mở rộng vaccine terms sang bệnh cụ thể.
    Sau đó chạy alias_map + rule-based để bắt thêm bệnh/triệu chứng.
    """
    text = f"{title}. {body}".strip()

    # Bước 1: expand vaccine terms
    vaccine_diseases = expand_vaccine_terms(text)

    # Bước 2: candidate phrases từ spaCy + mapped ngrams
    counts, raw_phrases = candidate_phrases(text)

    candidates, seen = [], set()

    for phrase_norm, freq in counts.most_common():
        if is_noise(phrase_norm):
            continue

        mapped = map_to_aliases(phrase_norm)

        if not mapped and looks_like_disease_by_text(phrase_norm):
            if phrase_norm not in seen:
                candidates.append({"term": phrase_norm, "cui": None, "score": 90})
                seen.add(phrase_norm)
            continue

        for m in mapped:
            canonical = (m.get("canonical") or m.get("alias") or phrase_norm)
            cui = m.get("cui")
            if (cui and has_disease_sty(cui)) or looks_like_disease_by_text(canonical):
                if canonical not in seen:
                    score = int(m.get("score", 100))
                    candidates.append({"term": canonical, "cui": cui, "score": score})
                    seen.add(canonical)

    # Bước 3: fallback
    if not candidates:
        for p in raw_phrases[:10]:
            pn = normalize(p)
            if is_noise(pn):
                continue
            if looks_like_disease_by_text(pn) and pn not in seen:
                candidates.append({"term": pn, "cui": None, "score": 80})
                seen.add(pn)

    # Bước 4: dedupe + rank
    best = {}
    for c in candidates:
        k = c["term"]
        if k not in best or c["score"] > best[k]["score"]:
            best[k] = c
    results = sorted(best.values(), key=lambda x: x["score"], reverse=True)

    # Bước 5: hậu kỳ lọc sạch
    diseases = []
    diseases.extend(vaccine_diseases)

    for r in results[:TOP_K]:
        cleaned = clean_disease_term(r["term"])
        if cleaned:
            diseases.append(cleaned)

    diseases = list(dict.fromkeys(diseases))

    return {"post_id": post_id, "diseases": diseases}


# ---------------------------
# CSV READER
# ---------------------------
def read_csv_rows(path):
    rows_out = []
    with open(path, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader, start=1):
            pid   = row.get("post_id") or str(i)
            title = row.get("post_title") or ""
            content = row.get("post_content") or ""
            rows_out.append({"post_id": pid, "title": title, "content": content})
    return rows_out


def clean_and_filter_diseases(disease_list):
    """
    Lọc sạch danh sách bệnh: giữ lại bệnh canonical ngắn gọn.
    """
    clean_list = []
    blacklist = ["anastomosis", "bypass", "substances that", "internal", "procedure"]

    for term in disease_list:
        # FIX: Handle dict terms trước khi process
        if isinstance(term, dict):
            term = str(term.get("disease") or term.get("name") or term.get("term") or "")

        term_lower = str(term).lower()

        if any(bad in term_lower for bad in blacklist):
            continue
        if term_lower in ["disorder", "bleeding", "disease", "syndrome"]:
            continue
        if any(w in term_lower for w in ["history", "descriptor", "question", "score",
                                          "value", "record", "doctor"]):
            continue
        if len(term_lower.split()) <= 4:
            clean_list.append(str(term).strip())

    return clean_list


# ---------------------------
# RULE-BASED PRE-FILTER
# ---------------------------

# Vaccine component terms — không bao giờ là disease
VACCINE_COMPONENTS = {
    "toxoid", "adjuvant", "antigen", "pertactin", "hemagglutinin",
    "neuraminidase", "capsid", "attenuated", "inactivated", "recombinant",
    "mrna", "vector", "lipid nanoparticle", "lnp", "spike protein",
    "formaldehyde", "aluminum", "thimerosal", "preservative",
    "booster", "dose", "series", "schedule", "immunoglobulin",
    "antibody", "titer", "serogroup", "serotype", "strain", "variant",
    "filamentous", "haemagglutin", "pertussis toxin", "fimbriae",
    "polysaccharide", "conjugate", "subunit", "live attenuated",
    "killed", "whole cell", "as01b", "as03", "mf59", "alum",
    "excipient", "diluent", "vial", "syringe", "injection site"
}

# Pattern tên vaccine trong text
VACCINE_NAME_PATTERNS = [
    r'\b(tdap|dtap|dtp|dtpa|mmr|mmrv)\b',
    r'\b(hpv|gardasil|cervarix)\b',
    r'\b(hep\s*[ab]|hepatitis\s*[ab]|heplisav|twinrix|engerix|recombivax)\b',
    r'\b(flu\s*shot|flu\s*vaccine|influenza\s*vaccine|fluzone|flucelvax|flublok)\b',
    r'\b(covid[-\s]?19\s*vaccine|mrna\s*vaccine|pfizer|moderna|astrazeneca|janssen|novavax)\b',
    r'\b(shingrix|zostavax|zoster\s*vaccine)\b',
    r'\b(prevnar|pneumovax|ppsv23|pcv\d+|vaxneuvance)\b',
    r'\b(vivotif|typhim|typhoid\s*vaccine)\b',
    r'\b(qdenga|dengvaxia|dengue\s*vaccine)\b',
    r'\b(nimenrix|menacwy|bexsero|trumenba|meningococcal\s*vaccine)\b',
    r'\b(varicella\s*vaccine|chickenpox\s*vaccine|varivax)\b',
    r'\b(rabies\s*vaccine|imovax|rabavert)\b',
    r'\b(yellow\s*fever\s*vaccine|stamaril|yf-vax)\b',
    r'\b(polio\s*vaccine|ipv|opv|imovax\s*polio)\b',
    r'\b(rotavirus\s*vaccine|rotateq|rotarix)\b',
    r'\b(acam2000|jynneos|mpox\s*vaccine|smallpox\s*vaccine)\b',
    r'\b(rsv\s*vaccine|abrysvo|arexvy)\b',
    r'\b(tdap|pertussis\s*vaccine|whooping\s*cough\s*vaccine)\b',
]

# Context patterns rõ ràng là DISEASE của patient
FIRST_PERSON_DISEASE_PATTERNS = [
    r'\b(i have|i got|i developed|i experienced|i am experiencing|i am suffering)\b',
    r'\b(experiencing|suffering from|diagnosed with|symptoms of)\b',
    r'\b(my (child|son|daughter|baby|kid|mom|dad|parent|partner|husband|wife) (has|had|developed|got))\b',
    r'\b(he has|she has|they have|he had|she had|they had)\b',
]


def is_vaccine_component_or_name(term: str) -> bool:
    """Check nếu term là vaccine component hoặc tên vaccine — không phải bệnh."""
    t = normalize(term)
    if not t:
        return False
    # Exact match với vaccine components
    if t in VACCINE_COMPONENTS:
        return True
    # Pattern match với tên vaccine
    for pattern in VACCINE_NAME_PATTERNS:
        if re.search(pattern, t, re.I):
            return True
    return False


def term_appears_in_disease_context(term: str, norm_text: str) -> bool:
    """
    Check xem term có xuất hiện trong context bệnh của patient không.
    Ví dụ: "I have fever" → True
           "fever vaccine" → False
    """
    t_escaped = re.escape(normalize(term))

    # Check first-person disease context
    for pattern in FIRST_PERSON_DISEASE_PATTERNS:
        combined = rf'{pattern}.*?{t_escaped}|{t_escaped}.*?{pattern}'
        if re.search(combined, norm_text, re.I | re.DOTALL):
            return True

    # Check nếu term xuất hiện ngay sau "have", "developed", "got" trong 5 từ
    proximity_pattern = (
        rf'\b(have|had|has|got|developed|experiencing|suffering from)\s+'
        rf'(?:\w+\s+){{0,4}}{t_escaped}'
    )
    if re.search(proximity_pattern, norm_text, re.I):
        return True

    return False


def term_only_in_vaccine_context(term: str, norm_text: str) -> bool:
    """
    Check nếu term CHỈ xuất hiện trong context tên vaccine.
    Ví dụ: "tetanus" chỉ trong "tetanus shot" → True
           "tetanus" trong "I have tetanus symptoms" → False
    """
    t_escaped = re.escape(normalize(term))
    vaccine_context = re.search(
        rf'{t_escaped}\s*(shot|vaccine|vaccination|jab|immunization|booster|dose|injection)|'
        rf'(shot|vaccine|vaccination|jab|immunization|booster|dose|injection)\s+.*?{t_escaped}',
        norm_text, re.I
    )
    return bool(vaccine_context)


def pre_filter_diseases(candidates: list, full_text: str) -> list:
    """
    Rule-based filter chạy TRƯỚC LLM.
    Loại bỏ những gì chắc chắn không phải disease của patient.
    Giảm workload cho LLM và tăng accuracy.
    """
    filtered = []
    norm_text = normalize(full_text)

    for term in candidates:
        # Handle dict input
        if isinstance(term, dict):
            term = str(term.get("disease") or term.get("name") or term.get("term") or "")

        t = normalize(str(term))
        if not t or len(t) < 3:
            continue

        # Rule 1: Loại vaccine components rõ ràng
        if is_vaccine_component_or_name(t):
            continue

        # Rule 2: Loại nếu term chỉ xuất hiện trong vaccine name context
        # và KHÔNG có trong first-person disease context
        if term_only_in_vaccine_context(t, norm_text):
            if not term_appears_in_disease_context(t, norm_text):
                continue

        # Rule 3: Loại generic terms
        if t in {"disease", "disorder", "syndrome", "condition", "infection",
                 "reaction", "symptom", "problem", "issue", "concern",
                 "complication", "side effect", "adverse event", "effect"}:
            continue

        # Rule 4: Loại nếu quá ngắn và không có suffix bệnh
        if len(t.split()) == 1 and len(t) < 4:
            if not any(t.endswith(suf) for suf in DISEASE_SUFFIXES):
                continue

        filtered.append(term)

    return filtered


# FIX 2: Updated system prompt — first-person disease filter
FIRST_PERSON_SYSTEM_PROMPT = (
    "You are a strict Medical Adjudicator.\n"
    "Rules:\n"
    "1. Keep ONLY diseases/symptoms the PATIENT (post author or their direct dependent "
    "such as their child or parent they care for) is ACTUALLY EXPERIENCING RIGHT NOW.\n"
    "2. EXCLUDE diseases that are ONLY mentioned as part of a vaccine name.\n"
    "   Examples to EXCLUDE:\n"
    "   - 'tetanus' from 'tetanus shot' or 'tetanus vaccine'\n"
    "   - 'hepatitis B' from 'Hep B vaccine'\n"
    "   - 'influenza' from 'flu vaccine' or 'flu shot'\n"
    "   - 'pertussis' from 'DTaP vaccine' or 'whooping cough vaccine'\n"
    "   - 'measles' from 'MMR vaccine'\n"
    "   - 'chickenpox' from 'varicella vaccine'\n"
    "3. EXCLUDE diseases of pets, animals, or any non-human subjects.\n"
    "4. EXCLUDE diseases mentioned in general medical discussion NOT about the poster.\n"
    "5. EXCLUDE diseases only mentioned as 'prevented by vaccine' without active symptoms.\n"
    "6. EXCLUDE hypothetical disease mentions ('vaccines can cause X', 'some people get X').\n"
    "7. INCLUDE only: active symptoms, diagnosed conditions, adverse reactions "
    "being experienced by the poster or their direct dependent.\n"
    "8. Output MUST be a raw JSON array of strings. Empty array [] if nothing qualifies.\n"
    "\n"
    "Decision examples:\n"
    "- 'I got a tetanus shot and now have arm pain' → [\"Arm pain\"]\n"
    "- 'I got the flu vaccine' → []\n"
    "- 'My cat has diarrhea after vaccination' → []\n"
    "- 'I developed myocarditis after COVID vaccine' → [\"Myocarditis\"]\n"
    "- 'Can vaccines cause autism?' → []\n"
    "- 'I have fever and chills after MMR shot' → [\"Fever\", \"Chills\"]\n"
    "- 'Study shows flu vaccine prevents influenza' → []\n"
    "- 'My child has rash and swelling at injection site' → [\"Rash\", \"Swelling\"]\n"
)


def llm_filter_candidates(text: str, candidates: list) -> list:
    """
    FIX: Updated với FIRST_PERSON_SYSTEM_PROMPT để filter đúng
    các trường hợp disease từ tên vaccine, thú cưng, hypothetical.
    """
    if not candidates:
        return []

    user_prompt = f"""
Post text: "{text}"
Candidate disease terms: {candidates}

Apply the rules strictly. Output ONLY a raw JSON array.
Result:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": FIRST_PERSON_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=150,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON array
        match = re.search(r'\[\s*(?:".*?"\s*,?\s*)*\]|\[\s*\]', content, re.DOTALL)
        content = match.group(0) if match else content

        parsed = json.loads(content)

        # Post-processing
        clean_results = []
        blacklist = {"diagnosis", "finding", "history", "patient", "doctor"}

        for d in parsed:
            # FIX: Handle dict trong parsed output
            if isinstance(d, dict):
                d = str(d.get("disease") or d.get("name") or d.get("term") or "")

            term = str(d).strip()
            term = re.sub(r'^(the patient has|patient has|symptoms include)\s+',
                          '', term, flags=re.I)

            t_low = term.lower()
            if t_low not in blacklist and 0 < len(term.split()) <= 5:
                term = re.sub(r'\s*\((disorder|diagnosis|finding)\)$', '',
                              term, flags=re.I)
                clean_results.append(term.strip().capitalize())

        return clean_results

    except Exception as e:
        print(f"DEBUG: LLM Filter error: {e}")
        return []


def safe_parse(content: str) -> list:
    start = content.find("[")
    if start == -1:
        return []
    snippet = content[start:]
    if "]" not in snippet:
        snippet = snippet + "]"
    try:
        return json.loads(snippet)
    except Exception as e:
        print("Parse error:", e)
        return []


def validate_icd_codes(disease_list: list, icd_code_list: list) -> list:
    """
    FIX: Post-processing ICD codes.
    1. Block severity qualifiers dùng standalone.
    2. Correct known wrong mappings.
    """
    validated = []
    for disease, code in zip(disease_list, icd_code_list):
        if not code or code == "NA":
            validated.append("NA")
            continue

        # Extract stem code
        stem = code.split(" ")[0].split("&")[0].strip()

        # Block standalone severity qualifiers
        if stem in ICD_BLACKLIST_STANDALONE:
            print(f"  [ICD FIX] Blocked standalone qualifier: {stem} for disease: {disease}")
            validated.append("NA")
            continue

        # Correct known wrong mappings
        if stem in ICD_CORRECTION_MAP:
            correct = ICD_CORRECTION_MAP[stem]
            print(f"  [ICD FIX] Corrected {stem} → {correct} for disease: {disease}")
            # Replace stem trong code string
            corrected_code = code.replace(stem, correct, 1)
            validated.append(corrected_code)
            continue

        validated.append(code)

    return validated


# ---------------------------
# LLM CLIENT
# ---------------------------
client     = OpenAI(base_url="http://192.168.0.215:1234/v1", api_key="not-needed")
MODEL_NAME = "llama-3.1-8b-instruct"


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def DiseaseCouncil(CSV_IN):
    rows = read_csv_rows(CSV_IN)
    print(f"DEBUG: read {len(rows)} rows from CSV")

    results = {}
    for row in rows:
        pid   = row.get("post_id") or ""
        title = row.get("title", "") or ""
        body  = row.get("content", "") or ""
        if not isinstance(body, str):
            body = str(body or "")

        # 1. Trích xuất ban đầu
        res = infer_for_row(pid, title, body)

        # 2. Lọc rule-based
        raw_diseases = res.get("diseases", [])
        filtered_diseases = clean_and_filter_diseases(raw_diseases)

        # 2b. Pre-filter rule-based trước LLM
        # Loại vaccine components, terms chỉ trong vaccine name context
        pre_filtered = pre_filter_diseases(filtered_diseases, f"{title}. {body}")

        # 3. LLM adjudication — chỉ xử lý những gì rule không bắt được
        final_diseases = llm_filter_candidates(f"{title}. {body}", pre_filtered)

        # 4. Lưu kết quả
        results[pid] = final_diseases
        print(f"  [{pid}] diseases: {final_diseases}")

    return results


if __name__ == "__main__":
    DiseaseCouncil(CSV_IN)