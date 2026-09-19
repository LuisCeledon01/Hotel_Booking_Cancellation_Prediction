"""Prediccion con el pipeline empaquetado."""
from __future__ import annotations

import functools
from typing import List, Union

import joblib
import numpy as np
import pandas as pd

from model import _RAIZ, __version__
from model.validation import validar

RUTA_PIPELINE = _RAIZ / "trained" / "pipeline.joblib"
UMBRAL_CLASE = 0.5  # umbral con el que se calcula `predictions` (el tablero aplica el suyo sobre `probabilities`)


@functools.lru_cache(maxsize=1)
def cargar_pipeline():
    if not RUTA_PIPELINE.exists():
        raise FileNotFoundError(f"No se encuentra el modelo empaquetado en {RUTA_PIPELINE}")
    return joblib.load(RUTA_PIPELINE)


def make_prediction(*, input_data: Union[pd.DataFrame, dict, List[dict]]) -> dict:
    """Devuelve {"predictions", "probabilities", "errors", "version"}."""
    if isinstance(input_data, dict):
        datos = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        datos = pd.DataFrame(input_data)
    else:
        datos = input_data.copy()

    validados, errores = validar(datos)
    resultado = {"predictions": None, "probabilities": None, "errors": errores, "version": __version__}
    if errores:
        return resultado

    proba = cargar_pipeline().predict_proba(validados)[:, 1]
    resultado["probabilities"] = np.asarray(proba, dtype=float)
    resultado["predictions"] = (proba >= UMBRAL_CLASE).astype(int)
    return resultado
