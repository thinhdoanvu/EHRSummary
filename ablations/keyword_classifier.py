"""
keyword_classifier.py
=====================
Keyword-only baseline classifier for vaccine misinformation detection.
Used as baseline comparison in Section 3.5 of:

  "Integrating ICD-11 with thematic taxonomies to classify mRNA vaccine
   misinformation for digital health surveillance"
   npj Digital Medicine (submission)

INPUT:  CSV file with column 'Raw Text / Summary' (AI-distilled post summary)
OUTPUT: CSV with binary classification (misinformation / neutral)

The classifier matches each post summary against a curated lexicon of 32
misinformation signal phrases spanning 7 semantic categories, with a
debunking guard of 6 phrases to exclude corrective/pro-vaccine contexts.

Reference standard: 972-post consensus ground truth derived from independent
annotation by two clinical domain experts (Expert 2 and Expert 3).

Results vs consensus GT (n=972):
  Precision: 42.0%  |  Recall: 32.2%  |  F1: 0.365  |  Accuracy: 79.2%
  TP=58  FP=80  FN=122  TN=712

Usage:
    python keyword_classifier.py --input gt_1000.csv --output keyword_results.csv
    python keyword_classifier.py --input gt_1000.csv --evaluate --gt_col consensus_label
"""

import os
import argparse
import pandas as pd
from collections import Counter

# ── SIGNAL PHRASE LEXICON ───────────────────────────────────────────────
# 32 phrases across 7 semantic categories

SIGNAL_PHRASES = {

    # Category 1: Vaccine-autism claims (4 phrases)
    "vaccine_autism": [
        "vaccines cause autism",
        "vaccine-autism",
        "autism diagnos",
        "wakefield",
    ],

    # Category 2: Toxic ingredient allegations (7 phrases)
    "toxic_ingredients": [
        "toxic spike protein",
        "spike protein",
        "aluminum adjuvant",
        "graphene",
        "toxic mrna",
        "mrna injection",
        "inflammatory cytokine",
    ],

    # Category 3: Safety study insufficiency claims (4 phrases)
    "safety_studies": [
        "lack of long-term safety",
        "no long-term studies",
        "long-term safety studies",
        "not been adequately tested",
    ],

    # Category 4: Data suppression signals (6 phrases)
    "data_suppression": [
        "authors have not disclosed",
        "hiding data",
        "retracted",
        "calls for retraction",
        "suppressed",
        "cover-up",
    ],

    # Category 5: Exaggerated harm claims (5 phrases)
    "exaggerated_harm": [
        "autoimmune disorder",
        "neurological damage",
        "adverse reaction",
        "death from vaccine",
        "persistent elevation",
    ],

    # Category 6: Genomic alteration tropes (4 phrases)
    "genomic_alteration": [
        "gene therapy",
        "alters dna",
        "changes dna",
        "excluded from trials",
    ],

    # Category 7: General misinformation labels (5 phrases)
    "general_labels": [
        "anti-vaccine",
        "antivax",
        "fake vaccine",
        "vaccine hoax",
        "conspiracy",
    ],
}

# All signal phrases as flat list
ALL_SIGNALS = [phrase for phrases in SIGNAL_PHRASES.values() for phrase in phrases]

# ── DEBUNKING GUARD ─────────────────────────────────────────────────────
# 6 phrases indicating corrective/pro-vaccine context
# If any debunking phrase is present, post is excluded from misinfo classification

DEBUNKING_PHRASES = [
    "retracted and discredited",
    "debunked",
    "falsified data",
    "no credible evidence",
    "scientific consensus",
    "disproven",
]

# ── CLASSIFIER ──────────────────────────────────────────────────────────

def classify_post(summary: str) -> dict:
    """
    Classify a single post summary.

    Returns dict with:
        - prediction: 'misinformation' or 'neutral'
        - matched_signal: first matched signal phrase (or None)
        - matched_category: semantic category of matched phrase (or None)
        - debunking_guard: True if debunking phrase blocked classification
    """
    text = str(summary or "").lower().strip()

    # Check debunking guard first
    for debunk in DEBUNKING_PHRASES:
        if debunk in text:
            return {
                "prediction":       "neutral",
                "matched_signal":   None,
                "matched_category": None,
                "debunking_guard":  True,
            }

    # Check signal phrases by category
    for category, phrases in SIGNAL_PHRASES.items():
        for phrase in phrases:
            if phrase in text:
                return {
                    "prediction":       "misinformation",
                    "matched_signal":   phrase,
                    "matched_category": category,
                    "debunking_guard":  False,
                }

    return {
        "prediction":       "neutral",
        "matched_signal":   None,
        "matched_category": None,
        "debunking_guard":  False,
    }


def classify_dataframe(df: pd.DataFrame,
                       text_col: str = "Raw Text / Summary") -> pd.DataFrame:
    """Classify all posts in a dataframe."""
    results = df[text_col].apply(classify_post).apply(pd.Series)
    return pd.concat([df, results], axis=1)


def evaluate(df: pd.DataFrame, gt_col: str = "consensus_label"):
    """Compute precision, recall, F1, accuracy against ground truth."""
    valid = df[df[gt_col].isin(["misinformation", "neutral"])].copy()
    n     = len(valid)

    y_true = valid[gt_col] == "misinformation"
    y_pred = valid["prediction"] == "misinformation"

    tp = (y_true  & y_pred).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true  & ~y_pred).sum()
    tn = (~y_true & ~y_pred).sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc  = (tp + tn) / n

    print(f"\n{'='*52}")
    print(f"EVALUATION vs '{gt_col}' (n={n})")
    print(f"{'='*52}")
    print(f"  GT misinfo:         {y_true.sum()} ({y_true.sum()/n*100:.1f}%)")
    print(f"  Predicted misinfo:  {y_pred.sum()} ({y_pred.sum()/n*100:.1f}%)")
    print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision : {prec*100:.1f}%")
    print(f"  Recall    : {rec*100:.1f}%")
    print(f"  F1        : {f1:.3f}")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  FP rate   : {fp/(fp+tn)*100:.1f}%  (of neutral posts)")
    print(f"  FN rate   : {fn/(fn+tp)*100:.1f}%  (of misinfo posts)")

    # False negative breakdown by narrative
    if "Specific Narrative" in valid.columns:
        fn_posts = valid[y_true & ~y_pred]
        print(f"\n  FN by narrative ({len(fn_posts)} posts):")
        for nar, cnt in Counter(fn_posts["Specific Narrative"]).most_common(8):
            if str(nar).strip() not in ("nan","None",""):
                print(f"    {nar}: {cnt}")

    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=round(prec, 4), recall=round(rec, 4),
                f1=round(f1, 4), accuracy=round(acc, 4))


# ── MAIN ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keyword-only vaccine misinformation classifier"
    )
    parser.add_argument("--input",    required=True,
                        help="Input CSV file path")
    parser.add_argument("--output",   default="keyword_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--text_col", default="Raw Text / Summary",
                        help="Column name containing post text/summary")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate against ground truth label")
    parser.add_argument("--gt_col",   default="consensus_label",
                        help="Ground truth column name (for --evaluate)")
    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.input, encoding="utf-8-sig", on_bad_lines="skip")
    print(f"Loaded: {len(df)} posts from {args.input}")
    print(f"Text column: '{args.text_col}'")

    # Classify
    df_out = classify_dataframe(df, text_col=args.text_col)
    n_mis  = (df_out["prediction"] == "misinformation").sum()
    print(f"\nClassified as misinformation: {n_mis} ({n_mis/len(df_out)*100:.1f}%)")
    print(f"Classified as neutral:        {len(df_out)-n_mis}")

    # Category breakdown
    cat_counts = df_out[df_out["prediction"]=="misinformation"]["matched_category"].value_counts()
    if len(cat_counts):
        print(f"\nMatched category breakdown:")
        for cat, cnt in cat_counts.items():
            print(f"  {cat}: {cnt}")

    # Evaluate if requested
    if args.evaluate and args.gt_col in df_out.columns:
        evaluate(df_out, gt_col=args.gt_col)

    # Save
    df_out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {args.output}")

    # Print lexicon summary
    print(f"\n{'─'*52}")
    print(f"LEXICON SUMMARY")
    print(f"{'─'*52}")
    print(f"  Total signal phrases: {len(ALL_SIGNALS)}")
    print(f"  Debunking phrases:    {len(DEBUNKING_PHRASES)}")
    for cat, phrases in SIGNAL_PHRASES.items():
        print(f"  {cat}: {len(phrases)} phrases")
