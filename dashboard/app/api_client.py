"""Cliente de la API de prediccion.

Contrato (ver CONTRATO_API.md):
  POST /api/v1/predict   {"inputs": [ {...reserva...}, ... ]}
    -> {"errors": null, "version": "...", "predictions": [0|1, ...],
        "probabilities": [0.0-1.0, ...]}
  GET  /api/v1/health

La direccion se toma de las variables de entorno API_URL y API_PORT, igual
que en el taller de Docker.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd
import requests

TIMEOUT = 30
LOTE = 500  # reservas por solicitud en los envios por lote

CAMPOS_OBLIGATORIOS = [
    "booking_date", "arrival_date", "lead_time", "stays_weekend", "stays_weekday",
    "adults", "meal_plan", "market_segment", "room_type", "deposit_type",
    "is_repeated_guest", "special_requests", "avg_daily_rate", "parking_spaces",
]
CAMPOS_OPCIONALES = [
    "booking_id", "children", "babies", "country", "agent_id",
    "previous_cancellations", "previous_bookings_not_canceled",
]
CAMPOS_API = CAMPOS_OBLIGATORIOS + CAMPOS_OPCIONALES
CAMPOS_ENTEROS = [
    "lead_time", "stays_weekend", "stays_weekday", "adults",
    "is_repeated_guest", "special_requests", "parking_spaces",
]


class ApiError(Exception):
    """Error de comunicacion o de validacion devuelto por la API."""


@dataclass
class Resultado:
    probabilidades: list[float]
    version: str
    tiene_probabilidades: bool


def base_url() -> str:
    host = os.getenv("API_URL", "localhost").strip()
    for prefijo in ("http://", "https://"):
        if host.startswith(prefijo):
            host = host[len(prefijo):]
    return f"http://{host.rstrip('/')}:{os.getenv('API_PORT', '8001')}"


def salud() -> dict | None:
    """Estado de la API, o None si no responde."""
    try:
        r = requests.get(f"{base_url()}/api/v1/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _a_registros(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> lista de dicts JSON-serializable (NaN pasa a null)."""
    faltan = [c for c in CAMPOS_OBLIGATORIOS if c not in df.columns]
    if faltan:
        raise ApiError("Al archivo le faltan columnas obligatorias: " + ", ".join(faltan))
    datos = df[[c for c in CAMPOS_API if c in df.columns]].copy()
    for col in CAMPOS_ENTEROS:
        datos[col] = pd.to_numeric(datos[col], errors="coerce").round().astype("Int64")
    for col in ("booking_date", "arrival_date"):
        datos[col] = pd.to_datetime(datos[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return json.loads(datos.astype(object).where(datos.notna(), None).to_json(orient="records"))


def _detalle(resp: requests.Response) -> str:
    try:
        detalle = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text[:300]
    if isinstance(detalle, list):
        partes = []
        for e in detalle[:5]:
            campo = ".".join(str(x) for x in e.get("loc", [])[-2:]) if isinstance(e, dict) else ""
            partes.append(f"{campo}: {e.get('msg', e)}" if isinstance(e, dict) else str(e))
        return "; ".join(partes)
    return str(detalle)


def predecir(df: pd.DataFrame) -> Resultado:
    """Envia las reservas a la API (en lotes) y devuelve las probabilidades."""
    registros = _a_registros(df)
    url = f"{base_url()}/api/v1/predict"
    probs: list[float] = []
    version = ""
    con_probs = True
    for i in range(0, len(registros), LOTE):
        try:
            resp = requests.post(url, json={"inputs": registros[i:i + LOTE]}, timeout=TIMEOUT)
        except requests.RequestException as exc:
            raise ApiError(f"No se pudo conectar con la API en {base_url()} ({type(exc).__name__}).") from exc
        if resp.status_code != 200:
            raise ApiError(f"La API respondió {resp.status_code}: {_detalle(resp)}")
        datos = resp.json()
        if datos.get("errors"):
            raise ApiError(f"La API reportó errores de validación: {datos['errors']}")
        version = datos.get("version", version)
        if datos.get("probabilities") is not None:
            probs += [float(p) for p in datos["probabilities"]]
        else:  # API sin probabilidades: se usa la clase (0/1)
            con_probs = False
            probs += [float(p) for p in datos["predictions"]]
    return Resultado(probs, version, con_probs)
