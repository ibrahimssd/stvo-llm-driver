# stvo-llm-driver

This project explores how well **Large Language Models (LLMs)** can understand and apply **German traffic regulations (StVO)** using driving license theory questions as benchmark tasks.

## Overview

The repository evaluates whether LLMs can answer theory-exam style questions correctly, and where they fail. The goal is to measure practical regulation understanding rather than generic language performance.

## What this project does

- Uses theoretical driving-license questions as test data.
- Prompts LLMs to answer these questions.
- Compares model answers against expected/correct outcomes.
- Produces pass-rate/accuracy style metrics for analysis.

## Why this matters

Traffic-law reasoning is a high-stakes domain where small mistakes matter. This project helps identify:

- Strengths of LLMs on rule-based reasoning.
- Common failure patterns (ambiguity, exceptions, edge cases).
- Readiness gaps for safety-critical use cases.

## Repository structure

> The exact file layout may evolve; the sections below describe the typical organization.

- `data/` – Driving theory questions and related assets.
- `src/` or main Python scripts – Evaluation and inference logic.
- `results/` – Model outputs, scores, or experiment summaries.
- `scripts/` – Shell utilities for running pipelines and experiments.

## Requirements

- Python 3.9+
- `pip` (or another Python package manager)
- (Optional) API credentials for the LLM provider(s) you test

## Setup

```bash
# 1) Clone
git clone https://github.com/ibrahimssd/stvo-llm-driver.git
cd stvo-llm-driver

# 2) (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate   # Windows PowerShell

# 3) Install dependencies
pip install -r requirements.txt
```

## Running experiments

If your repo already includes runnable scripts, use those directly. Typical patterns are:

```bash
# Example only – adjust to actual script names in this repo
python main.py
# or
python -m src.evaluate
# or
bash scripts/run_experiments.sh
```

## Outputs and evaluation

Typical outputs include:

- Per-question predictions.
- Aggregate metrics (e.g., accuracy/pass rate).
- Error analysis notes for incorrect answers.

When possible, report:

- Model name/version.
- Prompt format.
- Temperature and decoding settings.
- Dataset split/version.

## Reproducibility tips

- Pin dependency versions.
- Keep prompts and evaluation settings in version control.
- Save raw model outputs in addition to summary metrics.
- Set and document random seeds where relevant.

## Limitations

- Theory-question performance does **not** guarantee safe real-world driving behavior.
- Results can vary significantly by prompt design and model updates.
- Legal/regulatory interpretation can depend on context and jurisdiction updates.

## Contributing

Contributions are welcome. Good first contributions include:

- Better evaluation scripts and metrics.
- Improved dataset preprocessing.
- Prompting baselines and ablation studies.
- Documentation and reproducibility improvements.

## License

If not already specified, add a license file (e.g., `LICENSE`) to clarify usage rights.

## Citation

If you use this project in research or reports, please cite the repository and include commit hash/version details for reproducibility.
