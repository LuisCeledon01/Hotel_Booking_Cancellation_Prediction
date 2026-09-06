"""Transformadores de limpieza y feature engineering.

Cada transformador implementa la interfaz fit/transform de scikit-learn para
poder ir DENTRO de un Pipeline. Eso importa: si la imputacion o el encoding
se calculan sobre el train completo antes de partir en folds, los folds de
validacion ven informacion que no deberian ver y las metricas de validacion
cruzada salen optimistas. Metiendolo en el Pipeline, cada fold ajusta sus
propios estadisticos.

Las decisiones de limpieza vienen del EDA de la Entrega 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class _NamesOut:
    """Expone get_feature_names_out para que el Pipeline pueda nombrar las
    columnas de salida (necesario para reportar importancias de variables)."""

    def _record(self, X: pd.DataFrame) -> pd.DataFrame:
        self.feature_names_out_ = np.asarray(list(X.columns), dtype=object)
        return X

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "feature_names_out_"):
            raise AttributeError("Llame a fit/transform antes de get_feature_names_out")
        return self.feature_names_out_



class DropColumns(_NamesOut, BaseEstimator, TransformerMixin):
    """Elimina columnas (identificadores y fuga de informacion)."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        return self._record(X.drop(columns=[c for c in self.columns if c in X.columns]))


class DateFeatures(_NamesOut, BaseEstimator, TransformerMixin):
    """Deriva variables de calendario y elimina las fechas crudas.

    El EDA mostro estacionalidad leve (minimo ene-feb ~34%, pico oct ~42%) y
    que lead_time == arrival_date - booking_date en el 100% de las filas, asi
    que las fechas crudas no aportan nada que lead_time no tenga ya, salvo la
    estacionalidad. Se conservan mes, trimestre y dia de la semana de llegada.
    """

    def __init__(self, arrival: str = "arrival_date", booking: str = "booking_date"):
        self.arrival = arrival
        self.booking = booking

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.arrival in X.columns:
            arr = pd.to_datetime(X[self.arrival], errors="coerce")
            X["arrival_month"] = arr.dt.month
            X["arrival_quarter"] = arr.dt.quarter
            X["arrival_dayofweek"] = arr.dt.dayofweek
            X["arrival_is_weekend"] = (arr.dt.dayofweek >= 5).astype(int)
        X = X.drop(columns=[c for c in (self.arrival, self.booking) if c in X.columns])
        return self._record(X)


class DomainImputer(_NamesOut, BaseEstimator, TransformerMixin):
    """Imputa faltantes segun lo que significan, no con la media.

    - children (5.1% faltante): un faltante es "no se declararon menores".
      El valor correcto es 0, no el promedio de menores del dataset.
    - previous_cancellations (17.2% faltante): el EDA verifico que SOLO falta
      cuando is_repeated_guest == 0, es decir, huespedes sin historial previo.
      El valor correcto es 0. Imputar con la media le inventa un historial de
      cancelaciones a clientes nuevos.

    En ambos casos se agrega una bandera para que el modelo pueda distinguir
    un cero real de un cero imputado.
    """

    def __init__(self, zero_fill: list[str] | None = None, add_indicator: bool = True):
        self.zero_fill = zero_fill or ["children", "previous_cancellations"]
        self.add_indicator = add_indicator

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.zero_fill:
            if col not in X.columns:
                continue
            if self.add_indicator:
                X[f"{col}_was_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(0)
        return self._record(X)


class DomainFeatures(_NamesOut, BaseEstimator, TransformerMixin):
    """Variables derivadas del negocio."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if {"stays_weekend", "stays_weekday"}.issubset(X.columns):
            X["total_nights"] = X["stays_weekend"] + X["stays_weekday"]
            # El EDA encontro 133 reservas con 0 noches: reservas anomalas.
            X["zero_nights"] = (X["total_nights"] == 0).astype(int)
        if {"adults", "children", "babies"}.issubset(X.columns):
            X["total_guests"] = X["adults"] + X["children"].fillna(0) + X["babies"]
        if {"avg_daily_rate", "total_nights"}.issubset(X.columns):
            X["total_revenue"] = X["avg_daily_rate"] * X["total_nights"]
        if "special_requests" in X.columns:
            # Senal fuerte del EDA: 50% de cancelacion con 0 solicitudes.
            X["has_special_requests"] = (X["special_requests"] > 0).astype(int)
        if {"previous_cancellations", "previous_bookings_not_canceled"}.issubset(X.columns):
            prev_total = X["previous_cancellations"] + X["previous_bookings_not_canceled"]
            X["prev_cancel_ratio"] = np.where(
                prev_total > 0, X["previous_cancellations"] / prev_total, 0.0
            )
        return self._record(X)


class OutlierClipper(_NamesOut, BaseEstimator, TransformerMixin):
    """Recorta la cola superior de una variable continua (winsorizacion).

    El umbral se aprende SOLO con los datos de entrenamiento del fold.
    """

    def __init__(self, columns: list[str] | None = None, quantile: float = 0.99):
        self.columns = columns or ["avg_daily_rate"]
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y=None):
        self.upper_ = {
            c: float(X[c].quantile(self.quantile)) for c in self.columns if c in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, upper in getattr(self, "upper_", {}).items():
            if col in X.columns:
                X[col] = X[col].clip(upper=upper)
        return self._record(X)


class RareCategoryGrouper(_NamesOut, BaseEstimator, TransformerMixin):
    """Agrupa en 'RARE' las categorias por debajo de una frecuencia minima.

    country tiene 54 niveles y agent_id 50; muchos con un punado de reservas.
    Sin agrupar, el one-hot genera columnas casi vacias y el modelo memoriza
    ruido. Las categorias frecuentes se aprenden solo del fold de entrenamiento,
    de modo que una categoria no vista en test cae automaticamente en 'RARE'.
    """

    def __init__(self, columns: list[str] | None = None, threshold: float = 0.01):
        self.columns = columns or []
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        self.frequent_ = {}
        for col in self.columns:
            if col not in X.columns:
                continue
            freq = X[col].astype(str).value_counts(normalize=True)
            self.frequent_[col] = set(freq[freq >= self.threshold].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, keep in getattr(self, "frequent_", {}).items():
            if col in X.columns:
                X[col] = X[col].astype(str).where(X[col].astype(str).isin(keep), "RARE")
        return self._record(X)


class ToCategoricalString(_NamesOut, BaseEstimator, TransformerMixin):
    """Fuerza a texto las categoricas codificadas como enteros (agent_id).

    agent_id es un identificador, no una cantidad: que el agente 47 sea mayor
    que el 12 no significa nada. Los notebooks del repo le aplican LabelEncoder
    y lo dejan como numero, lo que le impone un orden inexistente.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype(str)
        return self._record(X)
