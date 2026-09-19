"""API SIMULADA para probar el tablero mientras la API real no esta lista.

Cumple el mismo contrato que la API definitiva (ver CONTRATO_API.md), pero la
probabilidad sale de una formula sencilla, NO del modelo entrenado. No usar en
produccion ni para reportar resultados.

    pip install fastapi "uvicorn<0.30"
    uvicorn tools.mock_api:app --port 8001          # desde la carpeta dashboard/

Luego, en otra terminal:

    set API_URL=localhost & set API_PORT=8001       (PowerShell: $env:API_URL="localhost"; $env:API_PORT="8001")
    cd app && streamlit run app.py
"""
from __future__ import annotations

import math
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

VERSION_MODELO = "mock-0.0.1"

app = FastAPI(title="HotelRisk API (simulada)")


class Reserva(BaseModel):
    booking_id: Optional[int] = None
    booking_date: str
    arrival_date: str
    lead_time: int
    stays_weekend: int
    stays_weekday: int
    adults: int
    children: Optional[float] = 0.0
    babies: Optional[float] = 0.0
    meal_plan: str
    country: Optional[str] = "Unknown"
    market_segment: str
    room_type: str
    deposit_type: str
    agent_id: Optional[float] = None
    is_repeated_guest: int
    previous_cancellations: Optional[float] = 0.0
    previous_bookings_not_canceled: Optional[float] = 0.0
    special_requests: int
    avg_daily_rate: float
    parking_spaces: int


def _probabilidad(r: Reserva) -> float:
    z = (
        -1.4
        + 0.011 * r.lead_time
        + 0.5 * (r.previous_cancellations or 0)
        - 0.3 * r.special_requests
        - 0.6 * r.is_repeated_guest
        + (0.6 if r.deposit_type == "No Deposit" else 0.0)
        - (0.9 if r.deposit_type == "Non Refund" else 0.0)
        + (0.3 if r.market_segment == "Online" else 0.0)
    )
    return 1 / (1 + math.exp(-z))


@app.get("/api/v1/health")
def health() -> dict:
    return {"name": "HotelRisk API (simulada)", "api_version": "0.0.1", "model_version": VERSION_MODELO}


@app.post("/api/v1/predict")
def predict(cuerpo: dict) -> dict:
    try:
        reservas = [Reserva(**r) for r in cuerpo.get("inputs", [])]
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors(include_url=False, include_input=False))
    probs = [round(_probabilidad(r), 4) for r in reservas]
    return {
        "errors": None,
        "version": VERSION_MODELO,
        "predictions": [int(p >= 0.5) for p in probs],
        "probabilities": probs,
    }
