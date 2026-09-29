# Agentic AI-Based Longitudinal EHR Summarisation for Multimorbidity Management in General Practice

A domain-partitioned, coordinator-mediated multi-agent pipeline that generates structured longitudinal EHR summaries directly from general practice (GP) tabular records, using the ACSQHC national guidelines as an organisational framework.

## Overview

Managing multimorbidity in general practice requires synthesising diagnostic, prescribing, and investigative information accumulated across years of fragmented clinical encounters. This repository contains the implementation of a pipeline that combines:

- **Rule-based extraction** (Stage 1) — deterministic extraction of eight structured sections directly from source fields
- **LLM-based synthesis** (Stage 2) — an ensemble of large language models generating three sections requiring longitudinal reasoning (Diagnoses, Clinical Summary, Recommendations)
- **Coordinator-mediated validation** (Stage 3) — cross-section consistency checking, evidence verification, and schema validation

The pipeline was evaluated on 2,232 longitudinal GP records from the electronic Practice-Based Research Network (ePBRN), South Western Sydney, Australia, including expert clinical review of 500 records and an ablation study comparing the proposed architecture against ten monolithic single-model variants.

## Repository Structure

```
.
├── generate_summary.py       # Main pipeline: rule-based extraction, LLM synthesis, coordinator validation
├── config.py                 # Model configuration, priority-ordered model pools, inference parameters
├── ablation_summary.py       # Ablation study runner (single-model variants A1–A10)
├── ablation_BC.py            # Additional ablation variants (coordinator/multi-model isolation)
├── ablation_metrics_bertscore.py  # BERTScore-based evaluation of ablation outputs
└── README.md
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) for local LLM inference
- Dependencies: `pandas`, `numpy`, `openai` (OpenAI-compatible client for Ollama), `scipy`, `scikit-learn`, `bert-score`, `tiktoken`

```bash
pip install pandas numpy openai scipy scikit-learn bert-score tiktoken
```

## Models Used

| Role | Priority-ordered model pool |
|---|---|
| Diagnoses / Clinical Summary / Recommendations | Llama 3.3 (70B), Qwen3 (32B), GPT-OSS (20B), Gemma 2 (27B), Llama 3 (8B) |
| Coordinator | Llama 3 (70B) |
| Rule-based support modules | Llama 3 (8B), Phi-4 (14B), Qwen3 (8B), Gemma 2 (27B), Llama 3.3 (70B) |

All inference was performed locally via [Ollama](https://ollama.com) on an NVIDIA RTX A6000 (96 GB VRAM). No patient data is transmitted to external services.

## Usage

### 1. Configure model pool and inference parameters

Edit `config.py` to set the Ollama server URL, chunk size, timeouts, and per-role model priority pools.

### 2. Generate summaries for the full cohort

```python
from generate_summary import generate_clinical_summary

generate_clinical_summary(
    file_input="path/to/patient_record.csv",
    file_output="path/to/output_summary.txt"
)
```

### 3. Run the ablation study

```bash
python ablation_summary.py
```

This replaces the multi-stage LLM pipeline with each of ten monolithic single-model variants while preserving the rule-based extraction layer, and evaluates output similarity against the proposed pipeline using BERTScore.

```bash
python ablation_metrics_bertscore.py
```

Computes Precision, Recall, and F1 (BERTScore, RoBERTa-large) between each ablation variant and the proposed pipeline output.

## Output Format

Each generated summary consists of eleven sections wrapped in structured markers:

```
[GROUP_START:SECTION_NAME]
...content...
[GROUP_END]
```

Sections without verifiable source evidence are emitted with a standardised `Not Recorded` placeholder rather than generating unsupported content — see the coordinator validation logic in `generate_summary.py` (schema validity criteria: structural marker integrity, mandatory field completeness, content validity).

## Data Availability

The ePBRN dataset used for evaluation is not publicly available due to patient privacy restrictions. Researchers wishing to access the dataset should contact the corresponding author to discuss data use agreements.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{Doan2026EHRSummarisation,
  author  = {Doan, Vu Thinh and others},
  title   = {Agentic AI Based Longitudinal EHR Summarisation for
             Multimorbidity Management in General Practice},
  journal = {Journal of the American Medical Informatics Association},
  year    = {2026}
}
```

## License

[Specify license, e.g., MIT / Apache 2.0]


