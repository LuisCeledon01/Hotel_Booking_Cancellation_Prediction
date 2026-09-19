"""Umbral de decision y bandas de riesgo del tablero.

Toda la logica que depende del umbral vive aqui, para que el tablero, los
reportes y las pruebas usen exactamente la misma regla.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ASSETS = Path(__file__).parent / "assets"

# Mismo valor que UMBRAL_RECOMENDADO en model/predict.py del repositorio.
# El F1 es casi plano entre 0,30 y 0,36; 0,35 es el que reporta el equipo
# en la tabla de la seccion 2.5.
UMBRAL_SUGERIDO = 0.35
UMBRAL_ESTANDAR = 0.50
UMBRAL_MIN, UMBRAL_MAX = 0.05, 0.95

# Desde aqui una reserva pasa de riesgo "Medio" a "Alto" (igual que CORTE_ALTO
# en model/predict.py). Si el umbral elegido es mayor, no hay banda Medio.
CORTE_ALTO = 0.60

COLORES = {"Bajo": "#106b57", "Medio": "#c9a227", "Alto": "#b5690a"}
FONDOS = {"Bajo": "#dcece6", "Medio": "#f7ecd9", "Alto": "#f7e2cf"}


def cargar_curva() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "umbral_curva.csv")


def metricas_en(curva: pd.DataFrame, umbral: float) -> dict:
    """Metricas de validacion cruzada en el umbral mas cercano de la curva."""
    idx = (curva["umbral"] - umbral).abs().idxmin()
    return curva.loc[idx].to_dict()


def corte_alto(umbral: float) -> float:
    return max(CORTE_ALTO, umbral)


def banda(prob: float, umbral: float) -> str:
    """Bajo: por debajo del umbral. Medio: entre el umbral y 0,60. Alto: desde 0,60."""
    if prob >= corte_alto(umbral):
        return "Alto"
    if prob >= umbral:
        return "Medio"
    return "Bajo"


def marcada(prob: float, umbral: float) -> bool:
    """True si la reserva se marca como probable cancelacion."""
    return prob >= umbral


def es(valor: float, decimales: int = 2) -> str:
    """Numero con coma decimal (0,35)."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def pct(valor: float, decimales: int = 0) -> str:
    return f"{valor * 100:.{decimales}f} %".replace(".", ",")
