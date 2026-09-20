"""Genera los datos que consume el tablero (dashboard/app/assets).

Se ejecuta UNA vez desde la raiz del repositorio, despues de `dvc pull`:

    python dashboard/scripts/build_assets.py

Asi el contenedor del tablero no necesita los datos crudos ni el modelo:
lleva solo agregados pequenos.

Produce:
  catalogos.json       categorias para los selectores del formulario
  analytics.json       tasas de cancelacion por segmento (pantalla Analytics)
  umbral_curva.csv     recall / precision / F1 por umbral, en pasos de 0,01
  ejemplo_reservas.csv 200 reservas de test.csv para probar el lote
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "app" / "assets"

# Campos que acepta la API (esquema HotelDataInputSchema del paquete del modelo)
API_FIELDS = [
    "booking_id", "booking_date", "arrival_date", "lead_time", "stays_weekend",
    "stays_weekday", "adults", "children", "babies", "meal_plan", "country",
    "market_segment", "room_type", "deposit_type", "agent_id", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled", "special_requests",
    "avg_daily_rate", "parking_spaces",
]


def build_catalogs(train: pd.DataFrame) -> dict:
    cat = {}
    for col in ["room_type", "deposit_type", "market_segment", "meal_plan"]:
        cat[col] = train[col].value_counts().index.tolist()
    cat["country"] = sorted(train["country"].dropna().unique().tolist())
    cat["country_default"] = train["country"].value_counts().index[0]
    cat["agent_id"] = sorted(int(a) for a in train["agent_id"].dropna().unique())
    return cat


def build_analytics(train: pd.DataFrame) -> dict:
    y = "is_canceled"

    def rate(col_or_series, name):
        g = train.groupby(col_or_series, observed=True)[y].agg(["mean", "size"]).reset_index()
        g.columns = [name, "tasa_cancelacion", "reservas"]
        return g.round(4).to_dict(orient="records")

    bins = [-1, 7, 30, 90, 180, np.inf]
    labels = ["0-7", "8-30", "31-90", "91-180", "181+"]
    lead = pd.cut(train["lead_time"], bins=bins, labels=labels)
    rep = train["is_repeated_guest"].map({0: "No", 1: "Sí"})
    return {
        "n_reservas": int(len(train)),
        "tasa_global": round(float(train[y].mean()), 4),
        "por_deposito": rate("deposit_type", "categoria"),
        "por_canal": rate("market_segment", "categoria"),
        "por_huesped_repetido": rate(rep.rename("huesped_repetido"), "categoria"),
        "por_antelacion": rate(lead.rename("antelacion_dias"), "categoria"),
    }


def build_curve(umbral_csv: Path, n_total: int) -> pd.DataFrame:
    """Interpola a pasos de 0,01 la tabla del equipo (pasos de 0,05)."""
    t = pd.read_csv(umbral_csv).sort_values("umbral")
    positivos = int((t["cancelaciones_detectadas"] + t["cancelaciones_perdidas"]).iloc[0])
    grid = np.round(np.arange(0.05, 0.9501, 0.01), 2)
    marc = np.interp(grid, t["umbral"], t["reservas_marcadas"])
    det = np.interp(grid, t["umbral"], t["cancelaciones_detectadas"])
    out = pd.DataFrame({"umbral": grid, "reservas_marcadas": marc, "cancelaciones_detectadas": det})
    out["falsas_alarmas"] = out["reservas_marcadas"] - out["cancelaciones_detectadas"]
    out["precision"] = out["cancelaciones_detectadas"] / out["reservas_marcadas"]
    out["recall"] = out["cancelaciones_detectadas"] / positivos
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    out["pct_marcadas"] = out["reservas_marcadas"] / n_total
    return out.round(4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    p.add_argument("--test", type=Path, default=ROOT / "data" / "test.csv")
    p.add_argument("--umbral", type=Path, default=ROOT / "reports" / "umbral_xgboost.csv")
    a = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(a.train)
    test = pd.read_csv(a.test)

    (OUT / "catalogos.json").write_text(json.dumps(build_catalogs(train), ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "analytics.json").write_text(json.dumps(build_analytics(train), ensure_ascii=False, indent=1), encoding="utf-8")
    build_curve(a.umbral, len(train)).to_csv(OUT / "umbral_curva.csv", index=False)
    test[[c for c in API_FIELDS if c in test.columns]].sample(200, random_state=42).to_csv(
        OUT / "ejemplo_reservas.csv", index=False
    )
    print(f"Assets escritos en {OUT}")


if __name__ == "__main__":
    main()
