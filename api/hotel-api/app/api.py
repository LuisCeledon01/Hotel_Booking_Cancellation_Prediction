from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger
from model import __version__ as model_version
from model import metadata
from model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


# Ruta para verificar que la API se esta ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    nombre = metadata().get("nombre", "modelo")
    return schemas.Health(
        name=settings.PROJECT_NAME,
        api_version=__version__,
        model_version=f"{nombre} {model_version}",
    ).model_dump()


# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """Probabilidad de cancelacion de una o varias reservas."""
    input_df = pd.DataFrame([r.model_dump() for r in input_data.inputs])

    logger.info(f"Prediccion para {len(input_df)} reserva(s)")
    resultados = make_prediction(input_data=input_df)

    if resultados["errors"] is not None:
        logger.warning(f"Error de validacion: {resultados['errors']}")
        raise HTTPException(status_code=400, detail=resultados["errors"])

    return {
        "errors": None,
        "version": resultados["version"],
        "predictions": resultados["predictions"].tolist(),
        "probabilities": resultados["probabilities"].tolist(),
    }
