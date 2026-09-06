# Cómo conectar el modelo al tablero

*Para Andrés. Cualquier duda me escribes. — Manuel*

El `.joblib` guarda el **pipeline completo** (limpieza + codificación + modelo),
así que el tablero no necesita saber nada del preprocesamiento: le pasas el
DataFrame tal como sale del CSV y te devuelve la probabilidad. No hay que
imputar, ni codificar, ni eliminar columnas antes.

## Integración mínima

```python
from model.predict import CancellationModel

modelo = CancellationModel()          # carga xgboost__clean, umbral 0.35
salida = modelo.predict(df_reservas)  # df crudo, con o sin is_canceled
```

`salida` es un DataFrame con:

| columna | contenido |
|---|---|
| `booking_id` | el identificador, si venía en la entrada |
| `probabilidad` | probabilidad de cancelación, 0 a 1 |
| `prediccion` | 1 si supera el umbral, 0 si no |
| `riesgo` | `Bajo` (<0,35), `Medio` (0,35–0,60), `Alto` (>0,60) |

Sobre `data/test.csv` (2.424 reservas) da: 527 de riesgo alto, 600 medio,
1.297 bajo.

## Si prefieres no importar Python en el tablero

```bash
python -m model.predict --input data/test.csv --output predicciones.csv
```

y el tablero lee `predicciones.csv`.

## Lo único que necesitas decidir: el umbral

El umbral por defecto de scikit-learn es 0,5 y **no es el adecuado aquí**. Lo
elegí en 0,35 maximizando F1 sobre las predicciones fuera de fold:

| umbral | reservas marcadas | cancelaciones detectadas | falsas alarmas | recall | precisión |
|---|---|---|---|---|---|
| 0,50 | 2.909 | 2.022 | 887 | 0,56 | 0,70 |
| **0,35** | 4.454 | 2.668 | 1.786 | **0,74** | 0,60 |
| 0,25 | 5.725 | 3.033 | 2.692 | 0,84 | 0,53 |

Si el tablero lo permite, **un control deslizante para el umbral sería el mejor
elemento de toda la interfaz**: le deja al usuario del negocio decidir cuántas
falsas alarmas está dispuesto a tolerar para no perderse una cancelación, que es
exactamente la pregunta de negocio del proyecto. Se cambia así:

```python
salida = modelo.predict(df_reservas, umbral=valor_del_slider)
```

El barrido completo está en `reports/umbral_xgboost.csv` y la gráfica en
`reports/umbral.png`, por si quieres mostrarla en el tablero.

## Otros modelos

```python
CancellationModel("lightgbm__clean")
CancellationModel("logistic_regression__clean")   # el más liviano, 15 KB
```

Los `.joblib` se generan corriendo `python -m model.train_pipeline`. No están
en el repo porque el de random forest pesa 19 MB.

## Advertencia

El modelo **no** usa `cancellation_fee_charged`: ese cargo se aplica después de
que el cliente cancela, así que en el momento de predecir no existe. Si el
tablero llega a mostrarla como variable de entrada, hay que quitarla.
