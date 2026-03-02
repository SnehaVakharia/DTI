# DTI Interaction Explorer (Minimal Pipeline)

Early-stage scaffold for a Drug Target Interaction (DTI) pipeline. This version is intentionally minimal: it only validates and accepts inputs, with no model inference or scoring.

## What exists right now

- A FastAPI backend with `/health`, `/ingest`, and `/status/{request_id}` endpoints.
- `/ingest` accepts SMILES + protein sequence and returns a request id.
- A simple vanilla JS UI to submit inputs and see pipeline status.

## Not included yet

- RDKit integration or production-grade model serving.
- Docker/production deployment.
- Real benchmark datasets bundled in this repo.

## Run locally

Backend:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m uvicorn app.main:app --reload --app-dir backend
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api` to `http://localhost:8000`.

## Baseline DTI Training (Simple CNN)

A reproducible baseline trainer is available at `backend/train_baseline_cnn.py`.
A one-command runner is available at `backend/run_baseline_cnn.sh`.

Expected CSV columns:

- `smiles`
- `protein_sequence`
- `label` (binary: `0` or `1`)

Install training dependencies:

```bash
source .venv/bin/activate
python -m pip install -r backend/requirements-train.txt
```

Run a reproducible DAVIS baseline:

```bash
bash backend/run_baseline_cnn.sh
```

Set `INSTALL_DEPS=1` if you want the script to run `pip install` first.

Run training manually on DAVIS:

```bash
source .venv/bin/activate
python backend/train_baseline_cnn.py \
  --prepare-davis-data \
  --data-path backend/data/davis_binary_dti.csv
```

Run training on synthetic data:

```bash
source .venv/bin/activate
python backend/train_baseline_cnn.py --generate-synthetic-data
```

Use your own dataset instead:

```bash
source .venv/bin/activate
python backend/train_baseline_cnn.py --data-path path/to/your_dti.csv
```

Generated deliverables:

- Baseline results table (CSV): `Deliverables/baseline_results.csv`
- Baseline results table (Markdown): `Deliverables/baseline_results.md`
- Latest model weights (convenience path): `Deliverables/models/baseline_cnn_weights.pt`
- Per-run model weights: `Deliverables/baseline_cnn_run/<run_id>/baseline_cnn_weights.pt`
- Per-run logs: `Deliverables/baseline_cnn_run/<run_id>/training_log.csv`
- Per-run metrics/config snapshot: `Deliverables/baseline_cnn_run/<run_id>/metrics.json`
<<<<<<< HEAD
=======

## Week4 Dataset Benchmark + Visualizations

This pipeline benchmarks **DAVIS**, **KIBA**, and **BindingDB_Kd** using the same simple CNN baseline,
saves clean splits, and exports slide-ready plots.

Run:

```bash
../.venv-dti/bin/python backend/run_week4_benchmark.py --max-samples 8000 --epochs 3
```

If you only want to regenerate plots/reports from existing runs:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp HOME=/tmp \
../.venv-dti/bin/python backend/generate_week4_visualizations.py
```

Key outputs:

- Dataset profiles: `Deliverables/week4/reports/dataset_profiles.csv`
- Baseline comparison table: `Deliverables/week4/reports/baseline_comparison.csv`
- Summary report: `Deliverables/week4/reports/summary.md`
- Saved splits: `Deliverables/week4/splits/<dataset>/{train,val,test}.csv`
- Model weights: `Deliverables/week4/models/*_baseline_cnn_weights.pt`
- Visualization directory (for PPT): `Deliverables/week4/visualizations/`
>>>>>>> a3bba6cf (Week 4 backend updates)
