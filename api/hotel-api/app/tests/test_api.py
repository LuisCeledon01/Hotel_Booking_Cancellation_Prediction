def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"name", "api_version", "model_version"}


def test_predict_devuelve_probabilidades(client, reserva):
    r = client.post("/api/v1/predict", json={"inputs": [reserva, reserva]})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["errors"] is None
    assert len(cuerpo["probabilities"]) == len(cuerpo["predictions"]) == 2
    assert all(0.0 <= p <= 1.0 for p in cuerpo["probabilities"])
    assert set(cuerpo["predictions"]) <= {0, 1}


def test_predict_ignora_columnas_de_fuga(client, reserva):
    r = client.post("/api/v1/predict", json={"inputs": [{**reserva, "cancellation_fee_charged": 1}]})
    assert r.status_code == 200


def test_predict_rechaza_datos_invalidos(client, reserva):
    malo = {k: v for k, v in reserva.items() if k != "room_type"}
    r = client.post("/api/v1/predict", json={"inputs": [malo]})
    assert r.status_code in (400, 422)
    assert "detail" in r.json()


def test_lote_con_faltantes_y_ceros(client, reserva):
    """Regresion: faltantes + cero cancelaciones/reservas previas no deben romper el pipeline."""
    filas = [
        {**reserva, "children": None, "previous_cancellations": None, "previous_bookings_not_canceled": 0},
        {**reserva, "children": 2, "previous_cancellations": 0, "previous_bookings_not_canceled": 0},
        {**reserva, "agent_id": 12, "previous_cancellations": 3, "previous_bookings_not_canceled": 2},
    ]
    r = client.post("/api/v1/predict", json={"inputs": filas})
    assert r.status_code == 200, r.text
    assert len(r.json()["probabilities"]) == 3


def test_coincide_con_el_pipeline_sobre_datos_crudos(client, reserva):
    """Regresion: la API debe dar lo mismo que el pipeline aplicado al CSV crudo.

    El pipeline convierte agent_id a texto y el modelo aprendio "12", no "12.0":
    si la API lo entregara como float, el agente se agruparia como RARE sin avisar.
    """
    import pandas as pd
    import pytest
    from model.predict import cargar_pipeline

    agente = 49  # agente frecuente en el entrenamiento (>1 %), no cae en RARE
    crudo = pd.DataFrame([{**reserva, "agent_id": agente, "children": 0.0, "previous_cancellations": 0.0}])
    esperado = float(cargar_pipeline().predict_proba(crudo)[:, 1][0])

    r = client.post("/api/v1/predict", json={"inputs": [{**reserva, "agent_id": agente,
                                                        "children": 0.0, "previous_cancellations": 0.0}]})
    assert r.json()["probabilities"][0] == pytest.approx(esperado)

    # y demuestra que el formato importa: con "49.0" el resultado seria otro
    crudo_mal = crudo.assign(agent_id=float(agente))
    assert float(cargar_pipeline().predict_proba(crudo_mal)[:, 1][0]) != pytest.approx(esperado)
