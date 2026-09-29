"""
single_llm_lmstudio.py
======================
Single-LLM baseline classifier using LMStudio (OpenAI-compatible API).
Mirrors single_llm_classifier.py interface for direct comparison.

INPUT:  CSV with 'Raw Text / Summary' column (same as keyword classifier)
OUTPUT: CSV with binary classification + metrics vs consensus GT

Safe models (not used in pipeline):
  ✅ openai/gpt-oss-120b   → recommended
  ✅ qwen3-235b-a22b       → largest, best if RAM allows

LMStudio server: http://192.168.0.215:1234/v1

Usage:
  # Single model
  python single_llm_lmstudio.py --model openai/gpt-oss-120b \\
      --input consensus_GT_expert_only.csv --evaluate

  # Test 10 posts first
  python single_llm_lmstudio.py --model openai/gpt-oss-120b \\
      --input consensus_GT_expert_only.csv --n 10 --evaluate

  # Run both safe models
  python single_llm_lmstudio.py --run-all \\
      --input consensus_GT_expert_only.csv --evaluate

  # Compare all results (Ollama + LMStudio)
  python single_llm_lmstudio.py --compare \\
      --input consensus_GT_expert_only.csv
"""

import os, re, sys, json, time, argparse
import pandas as pd
from pathlib import Path
from collections import Counter
from openai import OpenAI

# ── CONFIG ──────────────────────────────────────────────────────────────
LMSTUDIO_URL    = "http://192.168.0.215:1234/v1"
LMSTUDIO_APIKEY = "not-needed"
OUTPUT_DIR      = r"C:\Users\VU\Documents\NLP\Demo2\results"
TAXONOMY_JSON   = r"C:\Users\VU\Documents\NLP\Demo2\datasets\taxonomy.json"
TEMPERATURE     = 0.0
MAX_TOKENS      = 300
# ────────────────────────────────────────────────────────────────────────

SAFE_MODELS = [
    "openai/gpt-oss-120b",
    "qwen3-235b-a22b",
    "deepseek-r1-distill-llama-70b",
    "gemma-3-27b-it",
]


# ── Taxonomy ──────────────────────────────────────────────────────────────
def load_taxonomy() -> str:
    try:
        with open(TAXONOMY_JSON, encoding="utf-8") as f:
            tax = json.load(f)
        cfg   = tax["uns_taxonomy_config"]
        lines = ["NARRATIVE TAXONOMY:"]
        for sub in cfg["subtopics"]:
            lines.append(f"\nSubtopic: {sub['subtopic_name']}")
            for n in sub["narratives"]:
                lines.append(f"  - {n['specific_narrative']}: {n['description']}")
                lines.append(f"    Typical FLICC: {n['flicc']}")
        lines += [
            "\nFLICC TACTICS:",
            "  Exaggerated Harm | Fear Mongering | Conspiracy Theory |",
            "  Contextomy | Pseudoscience | Appeal to Emotion |",
            "  False Dichotomy | False Balance | None",
        ]
        return "\n".join(lines)
    except Exception:
        return ""


TAXONOMY_TEXT = load_taxonomy()

SYSTEM_PROMPT = f"""You are an expert in vaccine misinformation analysis.
Classify the following text using ONLY the taxonomy below.

{TAXONOMY_TEXT}

OUTPUT FORMAT — valid JSON only, no markdown, no explanation:
{{
  "subtopic": "Safety & Bodily Integrity | Efficacy & Necessity | Ethics & Liberties | None",
  "specific_narrative": "exact narrative name from taxonomy or None",
  "flicc": "FLICC tactic name or None",
  "disease": "disease names or NA",
  "icd11": "CODE (Label) or NA"
}}

STRICT RULES:
- Use ONLY narrative names listed above.
- Neutral health-seeking posts → all None.
- Classify as misinformation if the post promotes false or misleading vaccine claims.
- Output ONLY the JSON object."""


# ── API call ──────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    return OpenAI(base_url=LMSTUDIO_URL, api_key=LMSTUDIO_APIKEY)


def call_lmstudio(client: OpenAI, model: str,
                  text: str, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model       = model,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"TEXT:\n{text[:2000]}"},
                ],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
            content = resp.choices[0].message.content.strip()
            # Strip fences and thinking blocks
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$",     "", content).strip()
            content = re.sub(r"<think>.*?</think>", "", content,
                             flags=re.DOTALL).strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if {"subtopic","specific_narrative","flicc"}.issubset(data.keys()):
                    return data
        except Exception as e:
            print(f"    attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


# ── Helpers ───────────────────────────────────────────────────────────────
def clean(v):
    s = str(v or "").strip()
    return "None" if s in ("nan","NaN","","None","NA","N/A","none") else s


# ── Classify dataframe ────────────────────────────────────────────────────
def classify_dataframe(df: pd.DataFrame, client: OpenAI, model: str,
                       text_col: str = "Raw Text / Summary",
                       output_csv: str = None) -> pd.DataFrame:

    # Resume support
    existing_ids = set()
    cache = {}
    if output_csv and os.path.exists(output_csv):
        try:
            cached = pd.read_csv(output_csv, encoding="utf-8-sig")
            existing_ids = set(cached["Post ID"].astype(str).tolist())
            for _, row in cached.iterrows():
                cache[str(row["Post ID"])] = {
                    "pred_subtopic":  row.get("pred_subtopic",  "None"),
                    "pred_narrative": row.get("pred_narrative", "None"),
                    "pred_flicc":     row.get("pred_flicc",     "None"),
                    "pred_disease":   row.get("pred_disease",   "NA"),
                    "pred_icd11":     row.get("pred_icd11",     "NA"),
                    "is_misinfo":     row.get("is_misinfo",     False),
                }
            print(f"  Resuming — {len(existing_ids)} already classified")
        except Exception:
            pass

    records = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        post_id = str(row.get("Post ID", i)).strip()

        if post_id in existing_ids:
            records.append({**row.to_dict(), **cache[post_id]})
            continue

        text = str(row.get(text_col, "") or "").strip()
        if not text or text.lower() in ("nan", "none", ""):
            print(f"  [{i}/{total}] {post_id} — no text, skip")
            continue

        print(f"  [{i}/{total}] {post_id}...", end=" ", flush=True)
        result = call_lmstudio(client, model, text)

        if result:
            subtopic  = clean(result.get("subtopic",           "None"))
            narrative = clean(result.get("specific_narrative", "None"))
            flicc     = clean(result.get("flicc",              "None"))
            is_mis    = subtopic not in ("None","none","","NA")

            r = {
                "pred_subtopic":  subtopic,
                "pred_narrative": narrative,
                "pred_flicc":     flicc,
                "pred_disease":   str(result.get("disease","NA") or "NA"),
                "pred_icd11":     str(result.get("icd11",  "NA") or "NA"),
                "is_misinfo":     is_mis,
                "model":          model,
                "backend":        "lmstudio",
            }
            records.append({**row.to_dict(), **r})
            print(f"{'MISINFO' if is_mis else 'neutral'} | {narrative[:30]}")
        else:
            records.append({**row.to_dict(),
                             "pred_subtopic": "FAILED", "pred_narrative": "FAILED",
                             "pred_flicc": "FAILED", "pred_disease": "NA",
                             "pred_icd11": "NA", "is_misinfo": False,
                             "model": model, "backend": "lmstudio"})
            print("FAILED")

        if len(records) % 20 == 0 and output_csv:
            pd.DataFrame(records).to_csv(output_csv, index=False,
                                         encoding="utf-8-sig")
        time.sleep(0.3)

    result_df = pd.DataFrame(records)
    if output_csv:
        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result_df


# ── Evaluate ──────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, gt_col: str = "consensus_label",
             model_name: str = "") -> dict:
    valid = df[df[gt_col].isin(["misinformation","neutral"])].copy()
    valid = valid[valid["pred_subtopic"] != "FAILED"]
    n = len(valid)
    if n == 0:
        print("No valid rows to evaluate")
        return {}

    y_true = valid[gt_col] == "misinformation"
    y_pred = valid["is_misinfo"].astype(bool)

    tp  = (y_true  & y_pred).sum()
    fp  = (~y_true & y_pred).sum()
    fn  = (y_true  & ~y_pred).sum()
    tn  = (~y_true & ~y_pred).sum()
    pr  = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1  = 2*pr*rec/(pr+rec) if (pr+rec)>0 else 0
    acc = (tp+tn)/n
    fpr = fp/(fp+tn)*100 if (fp+tn)>0 else 0

    label = model_name or "Model"
    print(f"\n{'='*55}")
    print(f"{label} vs Consensus GT (n={n})")
    print(f"{'='*55}")
    print(f"  GT misinfo:         {y_true.sum()} ({y_true.sum()/n*100:.1f}%)")
    print(f"  Predicted misinfo:  {y_pred.sum()} ({y_pred.sum()/n*100:.1f}%)")
    print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision : {pr*100:.1f}%")
    print(f"  Recall    : {rec*100:.1f}%")
    print(f"  F1        : {f1:.3f}")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  FP rate   : {fpr:.1f}%")
    print(f"  FN rate   : {fn/(fn+tp)*100:.1f}%")

    fn_posts = valid[y_true & ~y_pred]
    if len(fn_posts) and "Specific Narrative" in valid.columns:
        print(f"\n  FN by narrative ({len(fn_posts)} posts):")
        for nar, cnt in Counter(fn_posts["Specific Narrative"]).most_common(8):
            if str(nar).strip() not in ("nan","None",""):
                print(f"    {nar}: {cnt}")

    return {"model": label, "backend": "lmstudio",
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": round(pr,4), "recall": round(rec,4),
            "f1": round(f1,4), "accuracy": round(acc,4),
            "fp_rate": round(fpr/100,4),
            "n_misinfo": int(y_pred.sum()), "n_total": n}


# ── Compare all ───────────────────────────────────────────────────────────
def compare_all(input_csv: str, gt_col: str = "consensus_label"):
    out_dir = Path(OUTPUT_DIR)
    files   = sorted(out_dir.glob("single_llm_*.csv"))
    if not files:
        print(f"No single_llm_*.csv files in {OUTPUT_DIR}")
        return

    gt_df = pd.read_csv(input_csv, encoding="utf-8-sig", on_bad_lines="skip")

    print(f"\n{'='*72}")
    print(f"COMPARISON TABLE — all systems vs Consensus GT")
    print(f"{'='*72}")
    print(f"  {'Model':<35} {'Backend':<10} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'Acc':>7} {'FP%':>6}")
    print(f"  {'-'*70}")

    all_results = []
    for fpath in files:
        df = pd.read_csv(fpath, encoding="utf-8-sig")
        df = df.merge(gt_df[["Post ID", gt_col]], on="Post ID",
                      how="inner", suffixes=("","_gt"))
        gt_use = f"{gt_col}_gt" if f"{gt_col}_gt" in df.columns else gt_col
        model_name = df["model"].iloc[0] if "model" in df.columns else fpath.stem
        backend    = df["backend"].iloc[0] if "backend" in df.columns else "ollama"

        valid = df[df[gt_use].isin(["misinformation","neutral"])].copy()
        valid = valid[valid.get("pred_subtopic","x").ne("FAILED")]
        if len(valid) == 0: continue

        y_true = valid[gt_use] == "misinformation"
        y_pred = valid["is_misinfo"].astype(bool)
        tp=(y_true & y_pred).sum(); fp=(~y_true & y_pred).sum()
        fn=(y_true & ~y_pred).sum(); tn=(~y_true & ~y_pred).sum()
        pr=tp/(tp+fp) if (tp+fp)>0 else 0
        rec=tp/(tp+fn) if (tp+fn)>0 else 0
        f1=2*pr*rec/(pr+rec) if (pr+rec)>0 else 0
        acc=(tp+tn)/len(valid)
        fpr=fp/(fp+tn)*100 if (fp+tn)>0 else 0

        print(f"  {model_name:<35} {backend:<10} {pr*100:>6.1f}% "
              f"{rec*100:>6.1f}% {f1*100:>6.1f}% {acc*100:>6.1f}% {fpr:>5.1f}%")
        all_results.append({"model": model_name, "backend": backend,
                             "precision": round(pr,4), "recall": round(rec,4),
                             "f1": round(f1,4), "accuracy": round(acc,4),
                             "fp_rate": round(fpr/100,4),
                             "tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn)})

    print(f"\n  {'--- Reference ---'}")
    print(f"  {'Keyword-only':<35} {'rule-based':<10} {'42.0%':>7} {'32.2%':>7} "
          f"{'36.5%':>7} {'79.2%':>7} {'58.0%':>6}")
    print(f"  {'Full pipeline':<35} {'multi-agent':<10} {'100.0%':>7} {'81.7%':>7} "
          f"{'89.9%':>7} {'96.6%':>7} {'0.0%':>6}")
    print(f"{'='*72}")

    compare_csv = os.path.join(OUTPUT_DIR, "comparison_all_systems.csv")
    pd.DataFrame(all_results).to_csv(compare_csv, index=False, encoding="utf-8-sig")
    print(f"\n📍 Saved: {compare_csv}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Single-LLM baseline classifier using LMStudio"
    )
    parser.add_argument("--input",    default="",
                        help="Input CSV file path")
    parser.add_argument("--model",    default="openai/gpt-oss-120b",
                        choices=SAFE_MODELS,
                        help="LMStudio model name")
    parser.add_argument("--text_col", default="Raw Text / Summary")
    parser.add_argument("--gt_col",   default="consensus_label")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--run-all",  action="store_true",
                        help="Run all safe LMStudio models")
    parser.add_argument("--compare",  action="store_true",
                        help="Compare all existing results")
    parser.add_argument("--n",        type=int, default=0,
                        help="Limit to first N posts (0 = all)")
    args = parser.parse_args()

    if args.compare:
        compare_all(args.input, args.gt_col)
        sys.exit(0)

    if not args.input:
        print("ERROR: --input required"); sys.exit(1)

    df = pd.read_csv(args.input, encoding="utf-8-sig", on_bad_lines="skip")
    if args.n > 0:
        df = df.head(args.n)
    print(f"Loaded: {len(df)} posts")
    print(f"LMStudio: {LMSTUDIO_URL}")

    client = get_client()
    models = SAFE_MODELS if args.run_all else [args.model]

    for model in models:
        safe_name  = model.replace("/","_").replace(":","_")
        output_csv = os.path.join(OUTPUT_DIR, f"single_llm_{safe_name}.csv")

        print(f"\n{'─'*55}")
        print(f"MODEL: {model}")
        print(f"{'─'*55}")

        df["model"]   = model
        df["backend"] = "lmstudio"
        result_df = classify_dataframe(df, client=client, model=model,
                                       text_col=args.text_col,
                                       output_csv=output_csv)

        n_mis = result_df["is_misinfo"].sum()
        print(f"\n  Classified misinfo: {n_mis} ({n_mis/len(result_df)*100:.1f}%)")

        if args.evaluate and args.gt_col in result_df.columns:
            evaluate(result_df, gt_col=args.gt_col, model_name=model)

    if args.run_all:
        compare_all(args.input, args.gt_col)

# RUN

# Test 10 posts trước
# python lmstudio_classifier.py --model openai/gpt-oss-120b --input results\consensus_GT_1000.csv --n 10 --evaluate

# Full 1000 posts: 1 model [openai/deepseek-r1-distill-llama-70b]
# python lmstudio_classifier.py --model openai/gpt-oss-120b --input results\consensus_GT_1000.csv --evaluate

# Cả 2 models
# python lmstudio_classifier.py --run-all --input results\consensus_GT_1000.csv --evaluate

#So sánh tất cả sau khi xong (cả Ollama + LMStudio): Dùng file LMStudio để compare — nó đọc tất cả single_llm_*.csv
# python lmstudio_classifier.py --compare --input results\consensus_GT_1000.csv
