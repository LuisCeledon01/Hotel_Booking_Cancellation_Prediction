"""Pruebas del preprocesamiento.

El objetivo no es cubrir todo, sino blindar las decisiones que se justificaron
en el EDA: que la fuga de informacion no vuelva a entrar, que los faltantes se
imputen con 0 y no con la media, y que el pipeline no aprenda nada del fold de
validacion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.pipeline import build_pipeline  # noqa: E402
from model.preprocessing import (  # noqa: E402
    DomainFeatures,
    DomainImputer,
    OutlierClipper,
    RareCategoryGrouper,
)
from model.train_pipeline import load_config  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def train(config):
    return pd.read_csv(ROOT / config["data"]["train_path"])


def test_imputa_con_cero_y_marca_bandera():
    df = pd.DataFrame({"children": [1.0, np.nan], "previous_cancellations": [np.nan, 2.0]})
    out = DomainImputer().fit_transform(df)
    assert out["children"].tolist() == [1.0, 0.0]
    assert out["previous_cancellations"].tolist() == [0.0, 2.0]
    assert out["children_was_missing"].tolist() == [0, 1]
    assert out["previous_cancellations_was_missing"].tolist() == [1, 0]


def test_previous_cancellations_solo_falta_en_huespedes_nuevos(train):
    """Justifica imputar con 0: quien nunca se ha hospedado no tiene historial."""
    faltantes = train[train["previous_cancellations"].isna()]
    assert len(faltantes) > 0
    assert (faltantes["is_repeated_guest"] == 0).all()


def test_agrupa_categorias_raras():
    df = pd.DataFrame({"country": ["ESP"] * 300 + ["AND", "MCO", "SMR"]})
    out = RareCategoryGrouper(["country"], threshold=0.01).fit_transform(df)
    assert set(out["country"]) == {"ESP", "RARE"}


def test_categoria_no_vista_cae_en_rare():
    grouper = RareCategoryGrouper(["country"], threshold=0.01).fit(
        pd.DataFrame({"country": ["ESP"] * 100})
    )
    out = grouper.transform(pd.DataFrame({"country": ["PRT"]}))
    assert out["country"].tolist() == ["RARE"]


def test_clipper_aprende_el_umbral_solo_del_train():
    clipper = OutlierClipper(["avg_daily_rate"], quantile=0.9)
    clipper.fit(pd.DataFrame({"avg_daily_rate": list(range(100))}))
    out = clipper.transform(pd.DataFrame({"avg_daily_rate": [5000.0]}))
    assert out["avg_daily_rate"].iloc[0] <= 100


def test_features_derivadas():
    df = pd.DataFrame(
        {
            "stays_weekend": [2],
            "stays_weekday": [0],
            "adults": [2],
            "children": [1.0],
            "babies": [0],
            "avg_daily_rate": [100.0],
            "special_requests": [0],
            "previous_cancellations": [1.0],
            "previous_bookings_not_canceled": [3],
        }
    )
    out = DomainFeatures().fit_transform(df)
    assert out["total_nights"].iloc[0] == 2
    assert out["total_guests"].iloc[0] == 3
    assert out["total_revenue"].iloc[0] == 200.0
    assert out["has_special_requests"].iloc[0] == 0
    assert out["prev_cancel_ratio"].iloc[0] == pytest.approx(0.25)


@pytest.mark.parametrize("mode", ["baseline", "clean"])
def test_la_fuga_de_informacion_nunca_llega_al_modelo(config, train, mode):
    """cancellation_fee_charged se cobra despues de cancelar: no puede entrar."""
    pipe = build_pipeline("logistic_regression", config, mode)
    X = train.drop(columns=[config["data"]["target"]]).head(500)
    y = train[config["data"]["target"]].head(500)
    pipe.fit(X, y)
    nombres = list(pipe.named_steps["preprocess"].get_feature_names_out())
    for col in config["leakage_columns"] + [config["data"]["id_column"]]:
        assert not any(col in n for n in nombres), f"{col} llego al modelo en modo {mode}"


def test_el_pipeline_no_ve_el_fold_de_validacion(config, train):
    """Un valor extremo solo en validacion no debe cambiar lo aprendido."""
    X = train.drop(columns=[config["data"]["target"]]).head(1000)
    y = train[config["data"]["target"]].head(1000)
    pipe = build_pipeline("logistic_regression", config, "clean")
    pipe.fit(X.iloc[:800], y.iloc[:800])
    umbral = pipe.named_steps["preprocess"].named_steps["clip"].upper_["avg_daily_rate"]

    X_contaminado = X.copy()
    X_contaminado.loc[X_contaminado.index[900], "avg_daily_rate"] = 1e6
    pipe2 = build_pipeline("logistic_regression", config, "clean")
    pipe2.fit(X_contaminado.iloc[:800], y.iloc[:800])
    assert pipe2.named_steps["preprocess"].named_steps["clip"].upper_["avg_daily_rate"] == umbral


def test_predice_probabilidades_validas(config, train):
    X = train.drop(columns=[config["data"]["target"]]).head(500)
    y = train[config["data"]["target"]].head(500)
    pipe = build_pipeline("xgboost", config, "clean")
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (500,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_maneja_test_sin_la_columna_objetivo(config):
    """El set de test no trae is_canceled: el pipeline debe correr igual."""
    train = pd.read_csv(ROOT / config["data"]["train_path"])
    test = pd.read_csv(ROOT / config["data"]["test_path"])
    pipe = build_pipeline("lightgbm", config, "clean")
    pipe.fit(train.drop(columns=[config["data"]["target"]]), train[config["data"]["target"]])
    proba = pipe.predict_proba(test)[:, 1]
    assert len(proba) == len(test)


def test_interfaz_de_prediccion_para_el_tablero(config, tmp_path):
    """El tablero recibe el CSV crudo y debe obtener probabilidad y banda."""
    from model.predict import CancellationModel
    from model.pipeline import build_pipeline
    import joblib

    train = pd.read_csv(ROOT / config["data"]["train_path"])
    pipe = build_pipeline("logistic_regression", config, "clean")
    pipe.fit(train.drop(columns=[config["data"]["target"]]), train[config["data"]["target"]])
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(pipe, ruta)

    modelo = CancellationModel(nombre="modelo", ruta=ruta)
    test = pd.read_csv(ROOT / config["data"]["test_path"])
    salida = modelo.predict(test)

    assert list(salida.columns) == ["booking_id", "probabilidad", "prediccion", "riesgo"]
    assert len(salida) == len(test)
    assert salida["probabilidad"].between(0, 1).all()
    assert set(salida["prediccion"].unique()) <= {0, 1}
    assert set(salida["riesgo"].dropna().unique()) <= {"Bajo", "Medio", "Alto"}


def test_prediccion_funciona_tambien_con_la_columna_objetivo(config, tmp_path):
    """Si le pasan el train completo (con is_canceled), no debe romperse."""
    from model.predict import CancellationModel
    from model.pipeline import build_pipeline
    import joblib

    train = pd.read_csv(ROOT / config["data"]["train_path"]).head(1000)
    pipe = build_pipeline("logistic_regression", config, "clean")
    pipe.fit(train.drop(columns=[config["data"]["target"]]), train[config["data"]["target"]])
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(pipe, ruta)

    salida = CancellationModel(nombre="modelo", ruta=ruta).predict(train)
    assert len(salida) == len(train)
