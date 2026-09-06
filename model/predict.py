"""Interfaz de prediccion para el tablero.

Pensado para que el tablero no tenga que saber nada del preprocesamiento: el
.joblib guarda el Pipeline completo (limpieza + modelo), asi que recibe el
DataFrame crudo tal como viene del CSV y devuelve la probabilidad.

Uso desde el tablero:

    from model.predict import CancellationModel

    modelo = CancellationModel()                  # carga xgboost__clean
    salida = modelo.predict(df_reservas)          # DataFrame con las columnas
                                                  # probabilidad, prediccion, riesgo

Uso por linea de comandos:

    python -m model.predict --input data/test.csv --output predicciones.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Umbral recomendado. NO es 0.5: se eligio maximizando F1 sobre las
# predicciones fuera de fold de la validacion cruzada. A 0.5 el modelo
# detecta el 56% de las cancelaciones; a 0.35 detecta el 74%, a cambio de
# mas falsas alarmas. Ver reports/umbral_xgboost.csv para el barrido completo
# y elegir otro valor si el costo del negocio lo justifica.
UMBRAL_RECOMENDADO = 0.35

# Cortes para las bandas de riesgo del tablero.
CORTE_MEDIO = 0.35
CORTE_ALTO = 0.60

MODELO_POR_DEFECTO = "xgboost__clean"


class CancellationModel:
    """Envoltura del pipeline entrenado."""

    def __init__(
        self,
        nombre: str = MODELO_POR_DEFECTO,
        umbral: float = UMBRAL_RECOMENDADO,
        ruta: Path | None = None,
    ):
        self.nombre = nombre
        self.umbral = umbral
        self.ruta = ruta or ROOT / "model" / "trained_models" / f"{nombre}.joblib"
        if not self.ruta.exists():
            raise FileNotFoundError(
                f"No existe {self.ruta}. Corra primero:\n"
                f"    python -m model.train_pipeline --models {nombre.split('__')[0]}"
            )
        self.pipeline = joblib.load(self.ruta)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Probabilidad de cancelacion para cada fila del DataFrame crudo."""
        datos = df.drop(columns=["is_canceled"], errors="ignore")
        proba = self.pipeline.predict_proba(datos)[:, 1]
        return pd.Series(proba, index=df.index, name="probabilidad")

    def predict(self, df: pd.DataFrame, umbral: float | None = None) -> pd.DataFrame:
        """Devuelve probabilidad, prediccion binaria y banda de riesgo.

        Conserva booking_id si viene, para poder cruzar con la tabla original.
        """
        umbral = self.umbral if umbral is None else umbral
        proba = self.predict_proba(df)
        salida = pd.DataFrame(index=df.index)
        if "booking_id" in df.columns:
            salida["booking_id"] = df["booking_id"].values
        salida["probabilidad"] = proba.round(4)
        salida["prediccion"] = (proba >= umbral).astype(int)
        salida["riesgo"] = pd.cut(
            proba,
            bins=[-0.001, CORTE_MEDIO, CORTE_ALTO, 1.001],
            labels=["Bajo", "Medio", "Alto"],
        )
        return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice cancelaciones sobre un CSV")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predicciones.csv"))
    parser.add_argument("--modelo", default=MODELO_POR_DEFECTO)
    parser.add_argument("--umbral", type=float, default=UMBRAL_RECOMENDADO)
    args = parser.parse_args()

    modelo = CancellationModel(args.modelo, args.umbral)
    df = pd.read_csv(args.input)
    salida = modelo.predict(df)
    salida.to_csv(args.output, index=False)

    print(f"Modelo:  {modelo.ruta.name}")
    print(f"Umbral:  {modelo.umbral}")
    print(f"Filas:   {len(salida)}")
    print(f"Marcadas como probable cancelacion: {int(salida.prediccion.sum())} "
          f"({salida.prediccion.mean():.1%})")
    print("\nDistribucion por banda de riesgo:")
    print(salida["riesgo"].value_counts().reindex(["Alto", "Medio", "Bajo"]).to_string())
    print(f"\nGuardado en {args.output}")


if __name__ == "__main__":
    main()
