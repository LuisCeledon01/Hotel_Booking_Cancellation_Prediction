"""Construccion de los pipelines de modelado.

Se definen dos configuraciones de preprocesamiento para poder medir cuanto
aporta la limpieza, en lugar de afirmarlo:

  - "baseline": replica el tratamiento de los notebooks iniciales del repo
    (imputacion por la media, codificacion ordinal de las categoricas, sin
    variables derivadas). Sirve como punto de comparacion.
  - "clean": el tratamiento derivado del EDA de la Entrega 1.

Ambas quitan cancellation_fee_charged: dejarla dentro no seria un baseline,
seria un modelo invalido.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from model.preprocessing import (
    DateFeatures,
    DomainFeatures,
    DomainImputer,
    DropColumns,
    OutlierClipper,
    RareCategoryGrouper,
    ToCategoricalString,
)


def _column_encoder(categorical: list[str], scale: bool) -> ColumnTransformer:
    """One-hot para las categoricas, passthrough (o escalado) para el resto."""
    from sklearn.preprocessing import FunctionTransformer

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            ("num", Pipeline(numeric_steps), _remaining(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class _remaining:
    """Selector de columnas: todo lo que no sea categorico."""

    def __init__(self, categorical: list[str]):
        self.categorical = categorical

    def __call__(self, X):
        return [c for c in X.columns if c not in self.categorical]


def build_preprocessor(config: dict, mode: str, scale: bool):
    """Devuelve el bloque de preprocesamiento completo para el modo indicado."""
    drop = list(config["leakage_columns"]) + [config["data"]["id_column"]]
    categorical = list(config["categorical_features"])

    if mode == "baseline":
        # Imputacion por la media + codificacion ordinal, como en los notebooks.
        return Pipeline(
            [
                ("drop", DropColumns(drop + ["booking_date", "arrival_date"])),
                ("to_str", ToCategoricalString(categorical)),
                (
                    "encode",
                    ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                Pipeline(
                                    [
                                        ("impute", SimpleImputer(strategy="most_frequent")),
                                        (
                                            "ordinal",
                                            OrdinalEncoder(
                                                handle_unknown="use_encoded_value",
                                                unknown_value=-1,
                                            ),
                                        ),
                                    ]
                                ),
                                categorical,
                            ),
                            (
                                "num",
                                Pipeline(
                                    [("impute", SimpleImputer(strategy="mean"))]
                                    + ([("scale", StandardScaler())] if scale else [])
                                ),
                                _remaining(categorical),
                            ),
                        ],
                        remainder="drop",
                        verbose_feature_names_out=False,
                    ),
                ),
            ]
        )

    if mode == "clean":
        return Pipeline(
            [
                ("drop", DropColumns(drop)),
                ("dates", DateFeatures()),
                ("impute_domain", DomainImputer()),
                ("features", DomainFeatures()),
                (
                    "clip",
                    OutlierClipper(["avg_daily_rate"], config["adr_clip_quantile"]),
                ),
                ("to_str", ToCategoricalString(categorical)),
                ("rare", RareCategoryGrouper(categorical, config["rare_threshold"])),
                ("encode", _column_encoder(categorical, scale)),
            ]
        )

    raise ValueError(f"modo de preprocesamiento desconocido: {mode}")


def get_estimator(name: str, random_state: int = 2025):
    """Devuelve (estimador, necesita_escalado)."""
    if name == "logistic_regression":
        return (
            LogisticRegression(max_iter=2000, C=1.0, random_state=random_state),
            True,
        )
    if name == "random_forest":
        return (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=random_state,
            ),
            False,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return (
            XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=random_state,
            ),
            False,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                n_jobs=-1,
                random_state=random_state,
                verbose=-1,
            ),
            False,
        )
    raise ValueError(f"modelo desconocido: {name}")


def build_pipeline(model_name: str, config: dict, mode: str = "clean") -> Pipeline:
    estimator, needs_scaling = get_estimator(model_name, config["cv"]["random_state"])
    return Pipeline(
        [
            ("preprocess", build_preprocessor(config, mode, scale=needs_scaling)),
            ("model", estimator),
        ]
    )
