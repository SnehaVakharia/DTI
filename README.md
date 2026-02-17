# DTI Interaction Explorer (Minimal Pipeline)

Early-stage scaffold for a Drug Target Interaction (DTI) pipeline. This version is intentionally minimal: it only validates and accepts inputs, with no model inference or scoring.

## What exists right now

- A FastAPI backend with `/health`, `/ingest`, and `/status/{request_id}` endpoints.
- `/ingest` accepts SMILES + protein sequence and returns a request id.
- A simple vanilla JS UI to submit inputs and see pipeline status.

## Not included yet

- Model training or RDKit integration.
- Scoring, plots, metrics, or monitoring.
- Docker/production deployment.
- Datasets or training notebooks.

## Run locally

Backend:

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api` to `http://localhost:8000`.
