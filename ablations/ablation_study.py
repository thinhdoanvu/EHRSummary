"""
ablation_study.py
=================
Ablation study trên 972 agreed GT posts — 4 conditions:

  A) no_summary   — Dùng raw post text thay vì distilled summary
  B) no_guards    — Chỉ Stage 1+2, bỏ guard stages 3-5
  C) no_iterative — Taxonomy 8 narratives (bỏ 3 narratives thêm sau)
  D) rag          — RAG: retrieve top-3 narratives thay vì inject 11

Data sources:
  - GT summaries:   consensus_GT_expert_detailed.csv (Raw Text / Summary)
  - Raw post text:  posts.csv (post_title + post_content) cho Condition A

Usage:
    python ablation_study.py --condition all
    python ablation_study.py --condition no_summary
    python ablation_study.py --condition no_guards
    python ablation_study.py --condition no_iterative
    python ablation_study.py --condition rag
"""

import sys, os, argparse, json, time, re
import pandas as pd
import numpy as np

# ── Chỉnh đường dẫn cho đúng máy ─────────────────────────────────
BACKEND_DIR   = r"C:\Users\VU\Documents\NLP\llm-council\backend"
PIPELINE_DIR  = r"C:\Users\VU\Documents\NLP\Demo2"
GT_LABELS_CSV = r"C:\Users\VU\Documents\NLP\Demo2\results\consensus_GT_expert_detailed.csv"
POSTS_CSV     = r"C:\Users\VU\Documents\NLP\Demo2\datasets\posts.csv"
TAXONOMY_FULL = r"C:\Users\VU\Documents\NLP\Demo2\datasets\taxonomy.json"
OUTPUT_DIR    = r"C:\Users\VU\Documents\NLP\Demo2\ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load config ───────────────────────────────────────────────────
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, PIPELINE_DIR)
try:
    from config import client, ModelManager
    print("✅ Config loaded.")
except ImportError as e:
    print(f"❌ Config error: {e}"); sys.exit(1)

from utils.NarrativeCouncil import NarrativeCouncil
from utils.RawtextSummaryl  import SummaryCouncil

# ── Narratives added iteratively ─────────────────────────────────
EXCLUDED_NARRATIVES = [
    "Ingredient Safety Concerns",
    "Suppression of Data",
    "Vaccine Safety and Efficacy",
]

# ── Helpers ───────────────────────────────────────────────────────
def is_misinfo(narr):
    return str(narr).strip().upper() not in [
        "", "NONE", "NAN", "N/A", "NA", "NULL", "FALSE", "-"
    ]

def compute_metrics(y_true, y_pred):
    TP  = sum(1 for t,p in zip(y_true,y_pred) if t and p)
    FP  = sum(1 for t,p in zip(y_true,y_pred) if not t and p)
    FN  = sum(1 for t,p in zip(y_true,y_pred) if t and not p)
    TN  = sum(1 for t,p in zip(y_true,y_pred) if not t and not p)
    P   = TP/(TP+FP) if (TP+FP)>0 else 0
    R   = TP/(TP+FN) if (TP+FN)>0 else 0
    F1  = 2*P*R/(P+R) if (P+R)>0 else 0
    ACC = (TP+TN)/len(y_true)
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,
            "Precision":round(P,3),"Recall":round(R,3),
            "F1":round(F1,3),"Accuracy":round(ACC,3)}

def load_taxonomy(path, exclude=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if exclude:
        for sub in data["uns_taxonomy_config"]["subtopics"]:
            sub["narratives"] = [
                n for n in sub["narratives"]
                if n["specific_narrative"] not in exclude
            ]
    n = sum(len(s["narratives"]) for s in data["uns_taxonomy_config"]["subtopics"])
    print(f"Taxonomy: {n} narratives loaded")
    return data

def load_gt(gt_path, posts_path):
    """
    Load GT labels + summaries + raw content.
    Returns list of dicts with: post_id, gt_misinfo, summary, raw_text
    """
    gt = pd.read_csv(gt_path, encoding="utf-8-sig", engine="python",
                     on_bad_lines="skip")
    gt["Post ID"] = gt["Post ID"].astype(str).str.strip()
    gt = gt[gt["annotator_agreement"] == "unanimous"].copy()
    gt["GT_misinfo"] = gt["consensus_label"] == "misinformation"

    posts = pd.read_csv(posts_path, encoding="utf-8-sig", engine="python",
                        on_bad_lines="skip")
    posts["post_id"] = posts["post_id"].astype(str).str.strip()

    merged = gt.merge(posts[["post_id","post_title","post_content"]],
                      left_on="Post ID", right_on="post_id", how="left")

    records = []
    for _, row in merged.iterrows():
        title   = str(row.get("post_title","")).strip()
        content = str(row.get("post_content","")).strip()
        raw = f"{title}\n\n{content}".strip() if content not in ("","nan") \
              else title
        records.append({
            "post_id":    row["Post ID"],
            "gt_misinfo": bool(row["GT_misinfo"]),
            "summary":    str(row.get("Raw Text / Summary","")).strip(),
            "raw_text":   raw if raw not in ("","nan") else "",
        })

    print(f"GT loaded: {len(records)} posts — "
          f"{sum(r['gt_misinfo'] for r in records)} misinfo, "
          f"{sum(not r['gt_misinfo'] for r in records)} neutral")
    has_raw = sum(1 for r in records if r["raw_text"])
    print(f"Posts with raw content: {has_raw}/{len(records)}")
    return records

def save_and_eval(results, condition_name):
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(OUTPUT_DIR, f"ablation_{condition_name}.csv"),
                  index=False, encoding="utf-8-sig")
    metrics = compute_metrics(df_out["gt_misinfo"].tolist(),
                              df_out["pred_misinfo"].tolist())
    print(f"\n[{condition_name}] {metrics}")
    return metrics

def classify_with_council(narrative_council, post_id, text):
    """Run NarrativeCouncil on text, return (pred_misinfo, narrative)."""
    try:
        result = narrative_council.process_narrative(post_id, text)
        narr   = result.get("Specific Narrative", "None")
        return is_misinfo(narr), narr
    except Exception as e:
        print(f"  [WARN] {post_id}: {e}")
        return False, "None"

def flat_classify(client, mm, system_prompt, summary, post_id):
    """Single LLM call with flat prompt — no council stages."""
    time.sleep(2.0)
    try:
        model = mm.get_model("CLIENT")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Summary: {summary[:600]}"},
            ],
            temperature=0, max_tokens=200, timeout=120
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        m   = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            narr = data.get("Narrative", "None")
            return is_misinfo(narr), narr
    except Exception as e:
        print(f"  [WARN] {post_id}: {e}")
    return False, "None"

def build_taxonomy_kb(taxonomy_data):
    """Build flat knowledge base string from taxonomy."""
    lines = []
    for sub in taxonomy_data["uns_taxonomy_config"]["subtopics"]:
        for nar in sub["narratives"]:
            lines.append(
                f"- Subtopic: {sub['subtopic_name']} | "
                f"Narrative: {nar['specific_narrative']} | "
                f"FLICC: {nar['flicc']} | "
                f"Stigma: {nar['stigma_target']} | "
                f"Trigger: {nar['real_world_trigger']}"
            )
    return "\n".join(lines)

# ── RAG index (in-memory semantic retrieval) ──────────────────────
def build_rag_index(taxonomy_data):
    narratives = []
    for sub in taxonomy_data["uns_taxonomy_config"]["subtopics"]:
        for nar in sub["narratives"]:
            text = (f"{nar['specific_narrative']} "
                    f"{nar.get('description','')} "
                    f"{' '.join(nar.get('keywords',[]))}")
            narratives.append({
                "subtopic":  sub["subtopic_name"],
                "narrative": nar["specific_narrative"],
                "flicc":     nar["flicc"],
                "stigma":    nar["stigma_target"],
                "trigger":   nar["real_world_trigger"],
                "text":      text,
            })
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([n["text"] for n in narratives],
                                   normalize_embeddings=True)
        print(f"✅ RAG index: SentenceTransformer ({len(narratives)} narratives)")
        return narratives, embeddings, model
    except ImportError:
        print("⚠️  sentence-transformers not found — using keyword overlap")
        return narratives, None, None

def retrieve_top_k(query, narratives, embeddings, model, k=3):
    if model is not None and embeddings is not None:
        from sentence_transformers import SentenceTransformer
        q_emb  = model.encode([query], normalize_embeddings=True)
        scores = (embeddings @ q_emb.T).flatten()
    else:
        query_words = set(query.lower().split())
        scores = np.array([
            len(query_words & set(n["text"].lower().split()))
            for n in narratives
        ], dtype=float)
    top_k = np.argsort(scores)[::-1][:k]
    return [narratives[i] for i in top_k]

def format_rag_kb(top_narratives):
    lines = []
    for n in top_narratives:
        lines.append(
            f"- Subtopic: {n['subtopic']} | Narrative: {n['narrative']} | "
            f"FLICC: {n['flicc']} | Stigma: {n['stigma']} | "
            f"Trigger: {n['trigger']}"
        )
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════
# CONDITION A — No SummaryCouncil (use raw text)
# ═══════════════════════════════════════════════════════════════════

def run_no_summary(records):
    print("\n" + "="*60)
    print("CONDITION A: No SummaryCouncil — raw post text → NarrativeCouncil")
    print("="*60)
    taxonomy = load_taxonomy(TAXONOMY_FULL)
    mm = ModelManager()
    narrative_council = NarrativeCouncil(client, mm, taxonomy)
    results = []

    for i, r in enumerate(records):
        # Use raw text; fallback to summary if no raw content
        text = r["raw_text"] if r["raw_text"] else r["summary"]
        text = text[:800]

        pred, narr = classify_with_council(narrative_council, r["post_id"], text)
        results.append({"post_id": r["post_id"], "gt_misinfo": r["gt_misinfo"],
                         "pred_misinfo": pred, "Narrative": narr})
        if (i+1) % 50 == 0:
            print(f"  [no_summary] {i+1}/{len(records)}")

    return save_and_eval(results, "no_summary")

# ═══════════════════════════════════════════════════════════════════
# CONDITION B — No Guard Stages (Stage 1 pre-screen + Stage 2 only)
# ═══════════════════════════════════════════════════════════════════

def run_no_guards(records):
    print("\n" + "="*60)
    print("CONDITION B: No Guard Stages — pre-screen + classify, no validation")
    print("="*60)
    taxonomy = load_taxonomy(TAXONOMY_FULL)
    mm = ModelManager()
    taxonomy_kb = build_taxonomy_kb(taxonomy)

    screen_sys = (
        "You are a fact-checker specializing in vaccine misinformation.\n"
        "Read the following vaccine-related summary.\n"
        "Answer ONLY 'YES' or 'NO': Does this contain vaccine MISINFORMATION, "
        "conspiracy theories, anti-vaccine rhetoric, or harmful false claims?\n"
        "Answer 'NO' for neutral questions, personal symptom reports, "
        "or factual vaccine information."
    )
    classify_sys = (
        "You are a Public Health Expert specializing in vaccine misinformation.\n"
        "Classify the summary using the taxonomy below.\n\n"
        f"TAXONOMY:\n{taxonomy_kb}\n\n"
        "Output ONLY JSON: {\"Subtopic\":\"...\",\"Narrative\":\"...\","
        "\"FLICC\":\"...\",\"Stigma\":\"...\",\"Trigger\":\"...\"}\n"
        "If no misinformation: set all fields to 'None'."
    )

    results = []
    model = mm.get_model("CLIENT")

    for i, r in enumerate(records):
        summary  = r["summary"][:600]
        narrative = "None"
        time.sleep(0.8)
        try:
            # Stage 1: pre-screen
            r1 = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":screen_sys},
                           {"role":"user","content":f"Summary: {summary}"}],
                temperature=0, max_tokens=5, timeout=60
            )
            answer = r1.choices[0].message.content.strip().upper()
            if "YES" in answer:
                # Stage 2: classify — NO guard stages after
                time.sleep(0.8)
                r2 = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":classify_sys},
                               {"role":"user","content":f"Summary: {summary}"}],
                    temperature=0, max_tokens=200, timeout=90
                )
                raw = r2.choices[0].message.content.strip()
                raw = re.sub(r'<think>.*?</think>','',raw,flags=re.DOTALL)
                m   = re.search(r'\{.*?\}', raw, re.DOTALL)
                if m:
                    narrative = json.loads(m.group(0)).get("Narrative","None")
        except Exception as e:
            print(f"  [WARN] {r['post_id']}: {e}")

        pred = is_misinfo(narrative)
        results.append({"post_id": r["post_id"], "gt_misinfo": r["gt_misinfo"],
                         "pred_misinfo": pred, "Narrative": narrative})
        if (i+1) % 50 == 0:
            print(f"  [no_guards] {i+1}/{len(records)}")

    return save_and_eval(results, "no_guards")

# ═══════════════════════════════════════════════════════════════════
# CONDITION C — No Iterative Taxonomy (8 narratives)
# ═══════════════════════════════════════════════════════════════════

def run_no_iterative(records):
    print("\n" + "="*60)
    print("CONDITION C: No Iterative Taxonomy — 8 narratives")
    print(f"Excluded: {EXCLUDED_NARRATIVES}")
    print("="*60)
    taxonomy_8 = load_taxonomy(TAXONOMY_FULL, exclude=EXCLUDED_NARRATIVES)
    mm = ModelManager()
    narrative_council = NarrativeCouncil(client, mm, taxonomy_8)
    results = []

    for i, r in enumerate(records):
        pred, narr = classify_with_council(
            narrative_council, r["post_id"], r["summary"])
        results.append({"post_id": r["post_id"], "gt_misinfo": r["gt_misinfo"],
                         "pred_misinfo": pred, "Narrative": narr})
        if (i+1) % 50 == 0:
            print(f"  [no_iterative] {i+1}/{len(records)}")

    return save_and_eval(results, "no_iterative")

# ═══════════════════════════════════════════════════════════════════
# CONDITION D — RAG-augmented (top-3 narrative retrieval)
# ═══════════════════════════════════════════════════════════════════

def run_rag(records):
    print("\n" + "="*60)
    print("CONDITION D: RAG — top-3 narrative retrieval per post")
    print("="*60)
    taxonomy = load_taxonomy(TAXONOMY_FULL)
    narratives, embeddings, embed_model = build_rag_index(taxonomy)
    mm = ModelManager()
    model = mm.get_model("CLIENT")

    screen_sys = (
        "You are a fact-checker. Answer ONLY 'YES' or 'NO': "
        "Does this summary contain vaccine MISINFORMATION, conspiracy theories, "
        "or anti-vaccine rhetoric?"
    )

    results = []
    for i, r in enumerate(records):
        summary  = r["summary"][:600]
        narrative = "None"
        time.sleep(0.8)
        try:
            # Pre-screen
            r1 = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":screen_sys},
                           {"role":"user","content":f"Summary: {summary}"}],
                temperature=0, max_tokens=5, timeout=90
            )
            answer = r1.choices[0].message.content.strip().upper()
            if "YES" in answer:
                # RAG: retrieve top-3 relevant narratives
                top_3 = retrieve_top_k(summary, narratives,
                                       embeddings, embed_model, k=3)
                rag_kb = format_rag_kb(top_3)
                classify_sys = (
                    "You are a Public Health Expert. Classify the summary "
                    "using ONLY these top-3 most relevant narratives:\n\n"
                    f"{rag_kb}\n\n"
                    "Output ONLY JSON: {\"Subtopic\":\"...\",\"Narrative\":\"...\","
                    "\"FLICC\":\"...\",\"Stigma\":\"...\",\"Trigger\":\"...\"}\n"
                    "If none match: set all to 'None'."
                )
                time.sleep(0.8)
                r2 = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":classify_sys},
                               {"role":"user","content":f"Summary: {summary}"}],
                    temperature=0, max_tokens=200, timeout=60
                )
                raw = r2.choices[0].message.content.strip()
                raw = re.sub(r'<think>.*?</think>','',raw,flags=re.DOTALL)
                m   = re.search(r'\{.*?\}', raw, re.DOTALL)
                if m:
                    narrative = json.loads(m.group(0)).get("Narrative","None")
        except Exception as e:
            print(f"  [WARN] {r['post_id']}: {e}")

        pred = is_misinfo(narrative)
        results.append({"post_id": r["post_id"], "gt_misinfo": r["gt_misinfo"],
                         "pred_misinfo": pred, "Narrative": narrative})
        if (i+1) % 50 == 0:
            print(f"  [rag] {i+1}/{len(records)}")

    return save_and_eval(results, "rag")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════

def print_summary(full_metrics, ablation_metrics):
    rows = [
        ("Full pipeline (reference)",          full_metrics),
        ("(A) w/o SummaryCouncil",             ablation_metrics.get("no_summary",   {})),
        ("(B) w/o Guard stages",               ablation_metrics.get("no_guards",    {})),
        ("(C) w/o Iterative taxonomy",         ablation_metrics.get("no_iterative", {})),
        ("(D) RAG-augmented (top-3)",          ablation_metrics.get("rag",          {})),
    ]
    print("\n" + "="*72)
    print(f"{'System':<40} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
    print("-"*72)
    for name, m in rows:
        if m:
            print(f"{name:<40} {m['Precision']:>6.3f} {m['Recall']:>6.3f} "
                  f"{m['F1']:>6.3f} {m['Accuracy']:>6.3f}")
    print("="*72)

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition",
                        choices=["all","no_summary","no_guards",
                                 "no_iterative","rag"],
                        default="rag")
    args = parser.parse_args()

    # Reference metrics from full pipeline
    full_metrics = {"Precision":1.000,"Recall":0.806,
                    "F1":0.892,"Accuracy":0.964}

    # Load GT data
    records = load_gt(GT_LABELS_CSV, POSTS_CSV)

    ablation_metrics = {}

    if args.condition in ("all","no_summary"):
        ablation_metrics["no_summary"]   = run_no_summary(records)
    if args.condition in ("all","no_guards"):
        ablation_metrics["no_guards"]    = run_no_guards(records)
    if args.condition in ("all","no_iterative"):
        ablation_metrics["no_iterative"] = run_no_iterative(records)
    if args.condition in ("all","rag"):
        ablation_metrics["rag"]          = run_rag(records)

    print_summary(full_metrics, ablation_metrics)

    rows = [{"Condition":c,**m} for c,m in ablation_metrics.items()]
    out  = os.path.join(OUTPUT_DIR, "ablation_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()