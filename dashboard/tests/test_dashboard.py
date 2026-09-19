"""Pruebas del tablero (no requieren la API ni el modelo).

    pip install pytest pandas
    pytest dashboard/tests -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import api_client  # noqa: E402
import umbral  # noqa: E402


def test_bandas_con_umbral_sugerido():
    t = umbral.UMBRAL_SUGERIDO
    assert umbral.banda(t - 0.01, t) == "Bajo"
    assert umbral.banda(t, t) == "Medio"
    assert umbral.banda(0.59, t) == "Medio"
    assert umbral.banda(0.60, t) == "Alto"


def test_con_umbral_alto_no_hay_banda_media():
    assert umbral.banda(0.70, 0.70) == "Alto"
    assert umbral.banda(0.69, 0.70) == "Bajo"


def test_marcada_solo_depende_del_umbral():
    assert umbral.marcada(0.40, 0.35) and not umbral.marcada(0.40, 0.50)


def test_curva_coincide_con_tabla_del_informe():
    """Valores de reports/umbral_xgboost.csv (seccion 2.5 del informe)."""
    curva = umbral.cargar_curva()
    m = umbral.metricas_en(curva, 0.50)
    assert round(m["reservas_marcadas"]) == 2910
    assert round(m["cancelaciones_detectadas"]) == 2005
    assert m["recall"] == pytest.approx(0.558, abs=0.001)
    assert umbral.metricas_en(curva, 0.35)["falsas_alarmas"] == pytest.approx(1795, abs=1)


def test_recall_baja_y_precision_sube_con_el_umbral():
    c = umbral.cargar_curva()
    assert c["recall"].is_monotonic_decreasing
    assert c["precision"].is_monotonic_increasing


def _reserva(**extra):
    base = {
        "booking_date": "2023-05-14", "arrival_date": "2023-06-24", "lead_time": 41, "stays_weekend": 4,
        "stays_weekday": 1, "adults": 2, "meal_plan": "Bed & Breakfast", "market_segment": "Corporate",
        "room_type": "Standard", "deposit_type": "No Deposit", "is_repeated_guest": 1,
        "special_requests": 1, "avg_daily_rate": 60.37, "parking_spaces": 0,
    }
    base.update(extra)
    return base


def test_registros_convierte_nan_en_null_y_quita_fuga():
    df = pd.DataFrame([_reserva(children=float("nan"), previous_cancellations=float("nan"),
                                cancellation_fee_charged=1, is_canceled=1)])
    reg = api_client._a_registros(df)[0]
    assert reg["children"] is None and reg["previous_cancellations"] is None
    assert "cancellation_fee_charged" not in reg and "is_canceled" not in reg
    assert isinstance(reg["lead_time"], int)


def test_registros_avisa_columnas_faltantes():
    with pytest.raises(api_client.ApiError, match="room_type"):
        api_client._a_registros(pd.DataFrame([_reserva()]).drop(columns=["room_type"]))


def test_base_url_acepta_host_con_o_sin_esquema(monkeypatch):
    monkeypatch.setenv("API_PORT", "8001")
    monkeypatch.setenv("API_URL", "http://1.2.3.4/")
    assert api_client.base_url() == "http://1.2.3.4:8001"
    monkeypatch.setenv("API_URL", "1.2.3.4")
    assert api_client.base_url() == "http://1.2.3.4:8001"
