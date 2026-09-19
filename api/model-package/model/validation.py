"""Esquema de entrada y validacion de reservas."""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

# Columnas que nunca llegan al modelo (fuga de informacion / identificador de la etiqueta)
COLUMNAS_PROHIBIDAS = ["cancellation_fee_charged", "is_canceled"]


class DataInputSchema(BaseModel):
    """Una reserva. Mismos campos que HotelDataInputSchema del paquete de la rama dev."""

    model_config = ConfigDict(extra="ignore")

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
    agent_id: Optional[int] = None
    is_repeated_guest: int
    previous_cancellations: Optional[float] = 0.0
    previous_bookings_not_canceled: Optional[float] = 0.0
    special_requests: int
    avg_daily_rate: float
    parking_spaces: int


# Columnas numericas opcionales: los faltantes deben ser NaN (float), no None (object),
# porque el pipeline hace aritmetica con ellas (por ejemplo cocientes con denominador 0).
COLUMNAS_NUMERICAS_OPCIONALES = [
    "children", "babies", "previous_cancellations", "previous_bookings_not_canceled",
]


def validar(datos: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[list]]:
    """Quita columnas prohibidas y valida cada fila. Devuelve (datos, errores|None)."""
    datos = datos.drop(columns=[c for c in COLUMNAS_PROHIBIDAS if c in datos.columns]).copy()
    for col in COLUMNAS_NUMERICAS_OPCIONALES:
        if col in datos.columns:
            datos[col] = pd.to_numeric(datos[col], errors="coerce").astype(float)
    # agent_id es un identificador: el pipeline lo convierte a texto y el modelo aprendio "28",
    # no "28.0". Si llegara como float (por un NaN en el lote), se agruparia como RARE sin avisar.
    if "agent_id" in datos.columns:
        datos["agent_id"] = pd.to_numeric(datos["agent_id"], errors="coerce").round().astype("Int64")
    registros = datos.replace({np.nan: None}).to_dict(orient="records")
    try:
        TypeAdapter(List[DataInputSchema]).validate_python(registros)
    except ValidationError as exc:
        return datos, json.loads(exc.json(include_url=False, include_input=False, include_context=False))
    return datos, None
