"""HotelRisk - tablero de riesgo de cancelacion de reservas.

Pantallas: Dashboard (consulta puntual), Analytics (panel historico) y
Reports (lote de reservas e historial de la sesion).

El tablero NO carga el modelo: envia cada reserva a la API de prediccion
(API_URL y API_PORT) y recibe la probabilidad de cancelacion. El umbral de
decision es un control del tablero (barra lateral) y se aplica sobre esa
probabilidad, de modo que cambiarlo no requiere volver a consultar la API.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client
from umbral import (
    COLORES, FONDOS, UMBRAL_ESTANDAR, UMBRAL_MAX, UMBRAL_MIN, UMBRAL_SUGERIDO,
    banda, cargar_curva, corte_alto, es, marcada, metricas_en, pct,
)

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="HotelRisk", page_icon="🏨", layout="wide")

# ---------------------------------------------------------------------
# Etiquetas en castellano. La API sigue recibiendo los valores originales
# (en ingles); estas etiquetas solo cambian lo que se ve en pantalla.
# ---------------------------------------------------------------------
ETIQUETAS = {
    "room_type": {"Standard": "Estándar", "Deluxe": "Deluxe", "Suite": "Suite"},
    "deposit_type": {"No Deposit": "Sin depósito", "Refundable": "Reembolsable", "Non Refund": "No reembolsable"},
    "market_segment": {"Online": "En línea", "Offline": "Fuera de línea", "Corporate": "Corporativo",
                       "Groups": "Grupos", "Aviation": "Aviación"},
    "meal_plan": {"Self Catering": "Solo alojamiento", "Half Board": "Media pensión",
                  "Bed & Breakfast": "Alojamiento y desayuno", "Full Board": "Pensión completa"},
}


def etiqueta(campo: str):
    return lambda valor: ETIQUETAS[campo].get(valor, valor)


@st.cache_data
def cargar_json(nombre: str) -> dict:
    return json.loads((ASSETS / nombre).read_text(encoding="utf-8"))


@st.cache_data
def curva() -> pd.DataFrame:
    return cargar_curva()


@st.cache_data(ttl=15)
def estado_api() -> dict | None:
    return api_client.salud()


CAT = cargar_json("catalogos.json")

# ---------------------------------------------------------------------
# Barra lateral: navegacion, umbral, estado de la API
# ---------------------------------------------------------------------
if "umbral" not in st.session_state:
    st.session_state["umbral"] = UMBRAL_SUGERIDO
if "historial" not in st.session_state:
    st.session_state["historial"] = []


def _restablecer_umbral() -> None:
    st.session_state["umbral"] = UMBRAL_SUGERIDO


st.sidebar.title("🏨 HotelRisk")
pagina = st.sidebar.radio("Ir a", ["Dashboard", "Analytics", "Reports"])

st.sidebar.divider()
st.sidebar.subheader("Umbral de decisión")
umbral = st.sidebar.slider(
    "Se marca como probable cancelación desde", UMBRAL_MIN, UMBRAL_MAX, step=0.01, key="umbral",
    help=(
        "Probabilidad a partir de la cual una reserva se marca como probable cancelación. "
        "Un umbral bajo detecta más cancelaciones pero genera más falsas alarmas; "
        "uno alto marca menos reservas, con mayor precisión."
    ),
)
st.sidebar.button(f"Restablecer a {es(UMBRAL_SUGERIDO)}", on_click=_restablecer_umbral)
_m = metricas_en(curva(), umbral)
st.sidebar.caption(
    f"Con este umbral, sobre las reservas históricas: se marca el {pct(_m['pct_marcadas'])}, "
    f"se detecta el {pct(_m['recall'])} de las cancelaciones (recall) y el "
    f"{pct(_m['precision'])} de las marcadas sí cancela (precisión). "
    "Estimado con XGBoost y validación cruzada."
)

st.sidebar.divider()
_salud = estado_api()
if _salud:
    st.sidebar.success(f"API conectada · modelo {_salud.get('model_version', '?')}")
else:
    st.sidebar.error("API sin respuesta")
st.sidebar.caption(f"Dirección: {api_client.base_url()}")


# ---------------------------------------------------------------------
# Sugerencias por banda de riesgo
# ---------------------------------------------------------------------
def sugerencia(nivel: str, tipo_deposito: str) -> str:
    if nivel == "Alto":
        if tipo_deposito != "Non Refund":
            return "Solicitar un depósito no reembolsable antes de confirmar la reserva."
        return "Contactar al huésped para confirmar la reserva u ofrecer flexibilidad de fecha."
    if nivel == "Medio":
        return "Enviar un recordatorio de confirmación cerca de la fecha de llegada."
    return "Reserva de bajo riesgo, no requiere acción adicional."


def medidor(prob: float, umbral: float, nivel: str) -> go.Figure:
    alto = corte_alto(umbral) * 100
    pasos = [
        {"range": [0, umbral * 100], "color": FONDOS["Bajo"]},
        {"range": [umbral * 100, alto], "color": FONDOS["Medio"]},
        {"range": [alto, 100], "color": FONDOS["Alto"]},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob * 100, 1), number={"suffix": " %"},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": COLORES[nivel]}, "steps": pasos,
            "threshold": {"line": {"color": "#222", "width": 3}, "thickness": 0.85, "value": umbral * 100},
        },
        title={"text": f"Riesgo: {nivel}"},
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=40, t=60, b=10))
    return fig


# =====================================================================
# Pantalla 1: Dashboard - consulta puntual
# =====================================================================
if pagina == "Dashboard":
    st.title("Nueva consulta de reserva")
    st.caption("Calcula el riesgo de cancelación de una reserva puntual con el modelo de la API.")

    zona_resultado = st.container()  # el resultado se dibuja arriba del formulario

    hoy = date.today()
    with st.form("consulta"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Estancia**")
            fecha_reserva = st.date_input("Fecha de la reserva", value=hoy)
            fecha_llegada = st.date_input("Fecha de llegada", value=hoy + timedelta(days=30))
            noches_semana = st.number_input("Noches entre semana", 0, 30, 3)
            noches_finde = st.number_input("Noches de fin de semana", 0, 10, 1)
            tipo_hab = st.selectbox("Tipo de habitación", CAT["room_type"], format_func=etiqueta("room_type"))
            plan = st.selectbox("Plan de alimentación", CAT["meal_plan"], format_func=etiqueta("meal_plan"))
        with c2:
            st.markdown("**Huésped**")
            pais = st.selectbox("País", CAT["country"], index=CAT["country"].index(CAT["country_default"]))
            adultos = st.number_input("Adultos", 0, 10, 2)
            ninos = st.number_input("Niños", 0, 10, 0)
            bebes = st.number_input("Bebés", 0, 5, 0)
            repetido = st.selectbox("Huésped repetido", ["No", "Sí"]) == "Sí"
            cancel_previas = st.slider("Cancelaciones previas del huésped", 0, 5, 0)
        with c3:
            st.markdown("**Comercial**")
            canal = st.selectbox("Canal de venta", CAT["market_segment"], format_func=etiqueta("market_segment"))
            deposito = st.selectbox("Tipo de depósito", CAT["deposit_type"], format_func=etiqueta("deposit_type"))
            tarifa = st.number_input("Tarifa promedio por noche (USD)", min_value=0.0, value=110.0, step=5.0)
            solicitudes = st.slider("Solicitudes especiales", 0, 5, 0)
            parqueos = st.number_input("Espacios de parqueo", 0, 5, 0)
        with st.expander("Opciones avanzadas"):
            a1, a2 = st.columns(2)
            agente = a1.selectbox("Agencia (agent_id)", [None] + CAT["agent_id"],
                                  format_func=lambda x: "Sin agencia" if x is None else str(x))
            reservas_previas = a2.number_input("Reservas previas no canceladas", 0, 50, 0)
        enviado = st.form_submit_button("Predecir riesgo de cancelación", type="primary")

    if enviado:
        if fecha_llegada < fecha_reserva:
            st.error("La fecha de llegada no puede ser anterior a la fecha de la reserva.")
        else:
            fila = pd.DataFrame([{
                "booking_date": fecha_reserva.isoformat(), "arrival_date": fecha_llegada.isoformat(),
                "lead_time": (fecha_llegada - fecha_reserva).days,
                "stays_weekend": noches_finde, "stays_weekday": noches_semana,
                "adults": adultos, "children": ninos, "babies": bebes,
                "meal_plan": plan, "country": pais, "market_segment": canal, "room_type": tipo_hab,
                "deposit_type": deposito, "agent_id": agente, "is_repeated_guest": int(repetido),
                "previous_cancellations": cancel_previas, "previous_bookings_not_canceled": reservas_previas,
                "special_requests": solicitudes, "avg_daily_rate": tarifa, "parking_spaces": parqueos,
            }])
            try:
                with st.spinner("Consultando la API..."):
                    res = api_client.predecir(fila)
                st.session_state["ultima"] = {
                    "prob": res.probabilidades[0], "con_probs": res.tiene_probabilidades,
                    "deposito": deposito, "pais": pais, "habitacion": tipo_hab,
                    "llegada": fecha_llegada.isoformat(), "antelacion": (fecha_llegada - fecha_reserva).days,
                }
                st.session_state["historial"].append({
                    "fecha_consulta": pd.Timestamp.now().floor("s"), "pais": pais,
                    "tipo_habitacion": tipo_hab, "llegada": fecha_llegada.isoformat(),
                    "probabilidad": res.probabilidades[0],
                })
            except api_client.ApiError as exc:
                st.session_state.pop("ultima", None)
                st.error(str(exc))

    # El resultado se dibuja desde el estado de la sesion: al mover el umbral
    # se actualiza al instante, sin volver a consultar la API.
    ultima = st.session_state.get("ultima")
    with zona_resultado:
        if ultima:
            prob = ultima["prob"]
            nivel = banda(prob, umbral)
            izq, der = st.columns([1.1, 1])
            with izq:
                st.plotly_chart(medidor(prob, umbral, nivel), key="medidor")
            with der:
                if not ultima["con_probs"]:
                    st.warning("La API no devolvió probabilidades, solo la clase (0/1): el umbral no tiene efecto.")
                st.metric("Probabilidad de cancelación", pct(prob, 1))
                if marcada(prob, umbral):
                    st.error(f"**Marcada como probable cancelación** (probabilidad ≥ umbral de {es(umbral)}).")
                else:
                    st.success(f"**No se marca como cancelación** (probabilidad < umbral de {es(umbral)}).")
                if marcada(prob, umbral) != marcada(prob, UMBRAL_ESTANDAR):
                    estandar = "se marcaría" if marcada(prob, UMBRAL_ESTANDAR) else "no se marcaría"
                    st.caption(f"Con el umbral estándar de {es(UMBRAL_ESTANDAR)} esta reserva {estandar}.")
                st.info(f"**Sugerencia:** {sugerencia(nivel, ultima['deposito'])}")
        else:
            st.caption("Completa el formulario y presiona **Predecir** para ver el resultado aquí.")

# =====================================================================
# Pantalla 2: Analytics - panel historico
# =====================================================================
elif pagina == "Analytics":
    datos = cargar_json("analytics.json")
    st.title("Panel histórico de reservas")
    st.caption("Agregados descriptivos del conjunto de entrenamiento y efecto del umbral de decisión.")

    k1, k2 = st.columns(2)
    k1.metric("Reservas analizadas", f"{datos['n_reservas']:,}".replace(",", "."))
    k2.metric("Tasa de cancelación global", pct(datos["tasa_global"], 1))


    def barras(clave: str, titulo: str, horizontal: bool, etiquetas: dict | None = None):
        df = pd.DataFrame(datos[clave])
        if etiquetas:
            df["categoria"] = df["categoria"].map(lambda v: etiquetas.get(v, v))
        if horizontal:
            df = df.sort_values("tasa_cancelacion")
            fig = px.bar(df, x="tasa_cancelacion", y="categoria", orientation="h", text_auto=".0%", title=titulo,
                         hover_data={"reservas": True})
            fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Tasa de cancelación", yaxis_title=None)
        else:
            fig = px.bar(df, x="categoria", y="tasa_cancelacion", text_auto=".0%", title=titulo,
                         hover_data={"reservas": True})
            fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Tasa de cancelación", xaxis_title=None)
        fig.update_traces(marker_color="#106b57")
        fig.update_layout(height=300)
        return fig


    f1c1, f1c2 = st.columns(2)
    f1c1.plotly_chart(barras("por_deposito", "Cancelación por tipo de depósito", True, ETIQUETAS["deposit_type"]), key="g1")
    f1c2.plotly_chart(barras("por_canal", "Cancelación por canal de venta", True, ETIQUETAS["market_segment"]), key="g2")
    f2c1, f2c2 = st.columns(2)
    f2c1.plotly_chart(barras("por_huesped_repetido", "Cancelación según huésped repetido", False), key="g3")
    lead = pd.DataFrame(datos["por_antelacion"])
    fig_lead = px.line(lead, x="categoria", y="tasa_cancelacion", markers=True,
                       title="Cancelación según antelación de la reserva (días)")
    fig_lead.update_layout(yaxis_tickformat=".0%", height=300, yaxis_title="Tasa de cancelación", xaxis_title=None)
    fig_lead.update_traces(line_color="#106b57")
    f2c2.plotly_chart(fig_lead, key="g4")

    st.subheader("Efecto del umbral de decisión")
    st.caption(
        "Cómo cambian el recall, la precisión y el F1 según el umbral (XGBoost, validación cruzada de 5 "
        "particiones sobre 9.696 reservas; interpolado a pasos de 0,01). La línea marca el umbral elegido "
        "en la barra lateral."
    )
    cv = curva()
    largo = cv.melt(id_vars="umbral", value_vars=["recall", "precision", "f1"], var_name="métrica", value_name="valor")
    largo["métrica"] = largo["métrica"].map({"recall": "Recall", "precision": "Precisión", "f1": "F1"})
    fig_u = px.line(largo, x="umbral", y="valor", color="métrica",
                    color_discrete_sequence=["#b5690a", "#106b57", "#555555"])
    fig_u.add_vline(x=umbral, line_width=2, line_dash="dash", line_color="#222")
    fig_u.update_layout(height=340, yaxis_tickformat=".0%", yaxis_title=None, xaxis_title="Umbral", legend_title=None)
    st.plotly_chart(fig_u, key="curva")

    m = metricas_en(cv, umbral)
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Reservas marcadas", f"{m['reservas_marcadas']:,.0f}".replace(",", "."), pct(m["pct_marcadas"]) + " del total", delta_color="off")
    u2.metric("Cancelaciones detectadas", f"{m['cancelaciones_detectadas']:,.0f}".replace(",", "."), pct(m["recall"]) + " (recall)", delta_color="off")
    u3.metric("Falsas alarmas", f"{m['falsas_alarmas']:,.0f}".replace(",", "."), f"precisión {pct(m['precision'])}", delta_color="off")
    u4.metric("F1", es(m["f1"], 3))

# =====================================================================
# Pantalla 3: Reports - lote de reservas e historial de la sesion
# =====================================================================
else:
    st.title("Reportes")
    tab_lote, tab_sesion = st.tabs(["Reservas por lote", "Consultas de esta sesión"])

    def tabla_resultado(df: pd.DataFrame, umbral: float, clave: str) -> pd.DataFrame:
        """Aplica el umbral actual y muestra filtros, indicadores y descarga."""
        df = df.copy()
        df["decision"] = df["probabilidad"].map(lambda p: "Marcar" if marcada(p, umbral) else "—")
        df["nivel_riesgo"] = df["probabilidad"].map(lambda p: banda(p, umbral))

        f1, f2, f3 = st.columns(3)
        paises = f1.multiselect("País", sorted(df["pais"].dropna().unique()), key=f"p_{clave}",
                                placeholder="Todos")
        habs = f2.multiselect("Tipo de habitación", sorted(df["tipo_habitacion"].dropna().unique()),
                              format_func=etiqueta("room_type"), key=f"h_{clave}", placeholder="Todos")
        niveles = f3.multiselect("Nivel de riesgo", ["Alto", "Medio", "Bajo"], key=f"n_{clave}", placeholder="Todos")
        if paises:
            df = df[df["pais"].isin(paises)]
        if habs:
            df = df[df["tipo_habitacion"].isin(habs)]
        if niveles:
            df = df[df["nivel_riesgo"].isin(niveles)]

        k1, k2, k3 = st.columns(3)
        k1.metric("Reservas", f"{len(df):,}".replace(",", "."))
        marc = int((df["decision"] == "Marcar").sum())
        k2.metric("Marcadas como probable cancelación", f"{marc:,}".replace(",", "."))
        k3.metric("% marcadas", pct(marc / len(df)) if len(df) else "—")

        vista = df.rename(columns={
            "booking_id": "Reserva", "pais": "País", "tipo_habitacion": "Habitación", "llegada": "Llegada",
            "fecha_consulta": "Consulta", "probabilidad": "Probabilidad", "decision": "Decisión",
            "nivel_riesgo": "Riesgo",
        })
        vista["Habitación"] = vista["Habitación"].map(etiqueta("room_type"))
        estilo = vista.style.map(lambda v: f"background-color: {FONDOS.get(v, '')}", subset=["Riesgo"]).format(
            {"Probabilidad": "{:.0%}"})
        st.dataframe(estilo, hide_index=True)
        st.download_button("Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"hotelrisk_{clave}.csv", mime="text/csv", key=f"d_{clave}")
        return df

    with tab_lote:
        st.caption(
            "Sube un CSV de reservas (mismas columnas que test.csv) o usa las reservas de ejemplo. "
            "Cada reserva se envía a la API una sola vez; luego el umbral de la barra lateral "
            "se puede cambiar sin volver a consultar."
        )
        b1, b2 = st.columns([2, 1])
        archivo = b1.file_uploader("Archivo CSV de reservas", type="csv")
        usar_ejemplo = b2.button("Usar 200 reservas de ejemplo")

        entrada, origen = None, ""
        if archivo is not None:
            entrada, origen = pd.read_csv(archivo), archivo.name
        elif usar_ejemplo:
            entrada, origen = pd.read_csv(ASSETS / "ejemplo_reservas.csv"), "ejemplo_reservas.csv"

        if entrada is not None:
            try:
                with st.spinner(f"Enviando {len(entrada)} reservas a la API..."):
                    res = api_client.predecir(entrada)
                lote = pd.DataFrame({
                    "booking_id": entrada["booking_id"].values if "booking_id" in entrada else range(len(entrada)),
                    "pais": entrada.get("country", pd.Series(["?"] * len(entrada))).values,
                    "tipo_habitacion": entrada["room_type"].values,
                    "llegada": entrada["arrival_date"].values,
                    "probabilidad": res.probabilidades,
                })
                st.session_state["lote"] = {"df": lote, "origen": origen, "con_probs": res.tiene_probabilidades,
                                            "version": res.version}
            except api_client.ApiError as exc:
                st.session_state.pop("lote", None)
                st.error(str(exc))

        lote = st.session_state.get("lote")
        if lote:
            st.markdown(f"**{lote['origen']}** · {len(lote['df'])} reservas · versión de la API: {lote['version'] or '—'}")
            if not lote["con_probs"]:
                st.warning("La API no devolvió probabilidades, solo la clase (0/1): el umbral no tiene efecto.")
            tabla_resultado(lote["df"], umbral, "lote")

    with tab_sesion:
        hist = st.session_state["historial"]
        if not hist:
            st.caption("Aún no hay consultas en esta sesión. Haz una en la pantalla Dashboard.")
        else:
            df_h = pd.DataFrame(hist).sort_values("fecha_consulta", ascending=False)
            tabla_resultado(df_h, umbral, "sesion")
