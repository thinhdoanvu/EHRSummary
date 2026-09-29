"""
vaccine_filter.py
=================
Pre-filter: chỉ giữ lại posts liên quan đến vaccine.
Dùng vaccine_to_diseases.json (đã có trong pipeline) làm source of truth.

Hai lớp:
  1. Rule-based — vaccine_to_diseases keys + brand names + generic terms
  2. LLM fallback — chỉ cho uncertain cases
"""

import re
import json
import os
import unicodedata

# ── Path đến vaccine_to_diseases.json ────────────────────────────────
VACCINE_JSON = r"C:\Users\VU\Documents\NLP\Demo2\datasets\UMLS_2024AA\vaccine_to_diseases.json"


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_vaccine_keywords(json_path: str) -> set:
    """
    Load vaccine keywords từ vaccine_to_diseases.json.
    Keys của file chính là vaccine names — dùng trực tiếp làm keywords.
    """
    if not os.path.exists(json_path):
        print(f"WARNING: vaccine_to_diseases.json not found: {json_path}")
        return set()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    keywords = set()

    # Keys = vaccine names từ file (dtap, tdap, flu, hpv, covid...)
    for vaccine_name in data.keys():
        keywords.add(normalize(vaccine_name))

    # Brand names và variants không có trong file nhưng phổ biến trên Reddit
    BRAND_VARIANTS = {
        # COVID
        "pfizer", "biontech", "moderna", "astrazeneca", "janssen",
        "johnson", "comirnaty", "spikevax", "covishield", "sinovac",
        "novavax", "nuvaxovid",
        # Flu variants
        "fluzone", "flucelvax", "flublok", "flulaval", "afluria",
        "influenza",
        # HPV brands
        "gardasil", "cervarix",
        # Hepatitis brands
        "engerix", "recombivax", "twinrix", "heplisav", "havrix", "vaqta",
        "hepatitis a", "hepatitis b",
        # Shingles brands
        "shingrix", "zostavax",
        # Pneumo brands
        "prevnar", "pneumovax", "ppsv23", "vaxneuvance",
        # Meningococcal brands
        "nimenrix", "bexsero", "trumenba", "menacwy", "menveo",
        # Varicella/MMR
        "varivax", "mmrv", "priorix",
        # Polio
        "ipv",
        # Rotavirus brands
        "rotateq", "rotarix",
        # Mpox/Smallpox
        "jynneos", "acam2000", "mpox",
        # RSV
        "abrysvo", "arexvy",
        # Dengue brands
        "qdenga", "dengvaxia",
        # Typhoid brands
        "vivotif", "typhim",
        # Rabies brands
        "rabavert",
        # Yellow fever brands
        "stamaril",
        # Generic vaccine terms
        "vaccination", "vaccinated", "unvaccinated", "vaccine",
        "immunization", "immunisation", "immunized", "immunised",
        "jab", "booster",
        "antivax", "anti-vax", "anti-vaccine", "pro-vaccine",
        "vaccine hesitancy", "vaccine mandate", "vaccine injury",
        "herd immunity", "vaccine efficacy", "vaccine safety",
        "mrna vaccine", "mrna",
        "vaccine schedule", "dose schedule",
        "second dose", "third dose", "fourth dose",
    }
    keywords.update(BRAND_VARIANTS)

    return keywords


# Load một lần khi import — tránh đọc file lặp lại
_VACCINE_KEYWORDS = None


def get_vaccine_keywords() -> set:
    global _VACCINE_KEYWORDS
    if _VACCINE_KEYWORDS is None:
        _VACCINE_KEYWORDS = load_vaccine_keywords(VACCINE_JSON)
    return _VACCINE_KEYWORDS


# Signals rõ ràng KHÔNG phải vaccine post
NON_VACCINE_SIGNALS = {
    "lobster", "otter", "crocodile", "beaver",
    "planet", "atmosphere", "telescope", "galaxy", "asteroid",
    "ant nest", "bee hive", "insect",
    "fishing", "aquaculture", "marine biology",
    "cryptocurrency", "bitcoin", "ethereum",
    "recipe", "cooking", "baking",
    "minecraft", "fortnite", "gaming",
    "stock market", "forex", "trading",
    "movie review", "song lyrics",
}


def is_vaccine_related_rule(title: str, body: str) -> tuple:
    """
    Rule-based vaccine relevance check.

    Returns:
        (bool, str) — (is_vaccine_related, confidence)
        confidence: 'high' | 'uncertain' | 'low'
    """
    text = normalize(f"{title} {body}")
    keywords = get_vaccine_keywords()

    # Nếu có non-vaccine signal VÀ không có vaccine keyword → block
    for sig in NON_VACCINE_SIGNALS:
        if sig in text:
            if not any(kw in text for kw in keywords):
                return False, 'low'

    # Check vaccine keywords từ file + brand variants
    for kw in keywords:
        if kw in text:
            return True, 'high'

    # Tier 2: injection + medical context → uncertain, nhờ LLM quyết
    has_injection = any(w in text for w in [
        'injection', 'injected', 'needle', 'syringe',
        'injection site', 'deltoid', 'intramuscular',
    ])
    has_medical = any(w in text for w in [
        'doctor', 'physician', 'clinic', 'cdc', 'who', 'nhs',
        'adverse reaction', 'side effect', 'immune', 'antibody',
    ])

    if has_injection and has_medical:
        return True, 'uncertain'

    return False, 'low'


def is_vaccine_related_llm(client, model_manager, title: str, body: str,
                            active_idx: int = 0) -> bool:
    """LLM confirmation cho uncertain cases."""
    import time

    text = f"{title}. {body}".strip()
    if len(text) > 500:
        text = text[:500] + "..."

    prompt = (
        "Answer ONLY 'YES' or 'NO'.\n"
        "Is the following Reddit post related to vaccines, vaccination, "
        "or vaccine-related health concerns?\n\n"
        f"Post: {text}\n\n"
        "Answer:"
    )

    model_name = model_manager.get_model("CLIENT", attempt=active_idx)
    if not model_name:
        return False

    try:
        time.sleep(1.0)
        params  = model_manager.get_params("CLIENT")
        timeout = model_manager.get_timeout(model_name)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **params,
            timeout=timeout,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"  [VaccineFilter] LLM error: {e}")
        return False


def should_process_post(client, model_manager, title: str, body: str,
                         use_llm_fallback: bool = True,
                         active_idx: int = 0) -> bool:
    """
    Main entry point: quyết định có xử lý post này không.
    Returns True nếu post liên quan vaccine.
    """
    is_related, confidence = is_vaccine_related_rule(title, body)

    if confidence == 'high':
        return True
    if confidence == 'low':
        return False

    # uncertain → hỏi LLM nếu được phép
    if use_llm_fallback:
        return is_vaccine_related_llm(
            client, model_manager, title, body, active_idx
        )

    return False  # conservative: skip nếu không dùng LLM