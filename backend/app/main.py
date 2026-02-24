from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MAX_SMILES_LENGTH = 2048
MAX_PROTEIN_LENGTH = 8000

app = FastAPI(title="DTI Analyser (Prototype)", version="0.0.0")


class InteractionRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=MAX_SMILES_LENGTH)
    protein_sequence: str = Field(..., min_length=1, max_length=MAX_PROTEIN_LENGTH)


class PipelineResponse(BaseModel):
    request_id: str
    status: str
    note: str


class PipelineStatus(BaseModel):
    request_id: str
    status: str
    received_at: str


PIPELINE_STORE: dict[str, PipelineStatus] = {}


def normalize_smiles(smiles: str) -> str:
    return "".join(smiles.split())


def normalize_protein(protein_sequence: str) -> str:
    return "".join(protein_sequence.split()).upper()


@app.get("/")
def index():
    return {
        "message": "Minimal DTI pipeline stub (no model output).",
        "endpoints": ["/health", "/ingest", "/status/{request_id}"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.post("/ingest", response_model=PipelineResponse)
def ingest(payload: InteractionRequest):
    if len(payload.smiles) > MAX_SMILES_LENGTH:
        raise HTTPException(status_code=400, detail="SMILES too long.")
    if len(payload.protein_sequence) > MAX_PROTEIN_LENGTH:
        raise HTTPException(status_code=400, detail="Protein sequence too long.")

    _ = normalize_smiles(payload.smiles)
    _ = normalize_protein(payload.protein_sequence)

    request_id = uuid4().hex
    PIPELINE_STORE[request_id] = PipelineStatus(
        request_id=request_id,
        status="accepted",
        received_at=datetime.now(timezone.utc).isoformat(),
    )
    return PipelineResponse(
        request_id=request_id,
        status="accepted",
        note="Input accepted. No model output is produced in this minimal pipeline.",
    )


@app.get("/status/{request_id}", response_model=PipelineStatus)
def status(request_id: str):
    stored = PIPELINE_STORE.get(request_id)
    if not stored:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    return stored
