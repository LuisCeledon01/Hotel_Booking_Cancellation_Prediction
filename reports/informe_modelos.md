# Modelos desarrollados y su evaluación

*Micro-proyecto — Entrega 2. Sección preparada por Manuel Linares (GitHub: ManoLord9408).*
*Todos los experimentos están registrados en el servidor MLflow del proyecto,
experimento `hotel_booking_cancellation`.*

## 1. Qué se entrenó y cómo

Se entrenaron **cuatro modelos** —regresión logística, random forest, XGBoost y
LightGBM— bajo **dos configuraciones de preprocesamiento**, para poder medir
cuánto aporta la limpieza de datos en lugar de afirmarlo:

| configuración | tratamiento |
|---|---|
| `baseline` | Réplica del tratamiento de los notebooks iniciales: imputación por la media, codificación ordinal de las categóricas, sin variables derivadas. |
| `clean` | Tratamiento derivado del EDA de la Entrega 1 (detalle en la sección 2). |

Las ocho combinaciones se evaluaron con **validación cruzada estratificada de 5
particiones** con la misma semilla, de modo que los folds son idénticos entre
configuraciones y la comparación es pareada.

Todo el preprocesamiento vive **dentro del `Pipeline` de scikit-learn**. Esto no
es un detalle de estilo: si los imputadores y codificadores se ajustan sobre el
train completo *antes* de partir en folds, cada fold de validación recibe
información calculada con sus propios datos y las métricas salen optimistas.
Metiéndolo en el pipeline, cada fold ajusta sus propios estadísticos.

### Exclusión de `cancellation_fee_charged`

La variable `cancellation_fee_charged` **se excluye de todos los modelos**. El
cargo por cancelación se aplica *después* de que el cliente cancela, así que en
el momento en que el modelo debe predecir no existe. En el EDA se midió que por
sí sola separa el objetivo casi perfectamente: 92,0 % de cancelación cuando vale
1 frente a 2,9 % cuando vale 0. Un modelo que la use reporta métricas
excelentes y es inservible en producción. Hay una prueba automatizada
(`test_la_fuga_de_informacion_nunca_llega_al_modelo`) que falla si vuelve a
colarse.

## 2. Selección y construcción de características

Cada decisión sale de un hallazgo del EDA:

| decisión | justificación |
|---|---|
| `previous_cancellations` faltante → **0 + bandera**, no la media | El 17,2 % de los registros no tiene valor, y se verificó que *todos* corresponden a `is_repeated_guest = 0`. Son huéspedes sin historial: el valor correcto es cero. Imputar con la media le inventa un historial de cancelaciones a clientes nuevos. |
| `children` faltante → **0 + bandera** | Un faltante significa "no se declararon menores". |
| Fechas crudas → **mes, trimestre, día de semana** de llegada | `lead_time` = `arrival_date − booking_date` en el 100 % de las filas, así que las fechas crudas son redundantes salvo por la estacionalidad (mínimo ene–feb ≈34 %, pico oct ≈42 %). |
| `country` y `agent_id`: **agrupar categorías con frecuencia < 1 % en "RARE"** y one-hot | 54 y 50 niveles respectivamente. Sin agrupar, el one-hot genera columnas casi vacías que el modelo memoriza. Además, una categoría no vista en test cae automáticamente en "RARE" en vez de romper el encoder. |
| `agent_id` tratado como **categoría, no como número** | Es un identificador. Que el agente 47 sea "mayor" que el 12 no significa nada; la codificación ordinal le impone un orden inexistente. |
| `avg_daily_rate`: **recorte al percentil 99** | El EDA halló 214 filas (~2 %) con tarifas de 900–1.500, fuera del rango del resto. Se recorta en vez de eliminar filas para no perder observaciones. El umbral se aprende solo con el fold de entrenamiento. |
| Variables derivadas | `total_nights`, `zero_nights` (133 reservas de 0 noches), `total_guests`, `total_revenue`, `has_special_requests`, `prev_cancel_ratio`. |

## 3. Resultados

Media de 5 folds. Métrica principal ROC-AUC; se reporta PR-AUC porque la clase
positiva es minoritaria (37,1 %) y el *accuracy* sería engañoso.

| modelo | preproc. | ROC-AUC | PR-AUC | F1 @0,5 | Recall @0,5 |
|---|---|---|---|---|---|
| XGBoost | **clean** | **0,7978** ± 0,0101 | **0,7113** | 0,6162 | 0,5577 |
| LightGBM | clean | 0,7872 ± 0,0095 | 0,6970 | 0,6111 | 0,5547 |
| Regresión logística | clean | 0,7855 ± 0,0106 | 0,6958 | 0,6003 | 0,5477 |
| Random forest | clean | 0,7843 ± 0,0073 | 0,6896 | 0,5151 | 0,3944 |
| XGBoost | baseline | 0,7890 ± 0,0130 | 0,7028 | 0,6066 | 0,5449 |
| LightGBM | baseline | 0,7800 ± 0,0118 | 0,6890 | 0,5973 | 0,5388 |
| Random forest | baseline | 0,7777 ± 0,0106 | 0,6887 | 0,5618 | 0,4662 |
| Regresión logística | baseline | 0,7406 ± 0,0077 | 0,6472 | 0,5409 | 0,4662 |

### Aporte de la limpieza (folds pareados)

| modelo | baseline | clean | diferencia | p (t pareado) |
|---|---|---|---|---|
| Regresión logística | 0,7406 | 0,7855 | **+0,0449** | 0,00001 |
| XGBoost | 0,7890 | 0,7978 | +0,0088 | 0,008 |
| LightGBM | 0,7800 | 0,7872 | +0,0072 | 0,009 |
| Random forest | 0,7777 | 0,7843 | +0,0066 | 0,049 |

La limpieza mejora los cuatro modelos y la mejora es consistente fold a fold.
**Dos advertencias honestas:** (a) en random forest el p-valor queda justo en el
límite de 0,05, así que ahí la evidencia es débil; (b) cinco folds son pocos y
comparten datos entre sí, por lo que el t pareado sobre folds es optimista
(Dietterich, 1998) — sirve para descartar que la mejora sea ruido, no como
prueba formal.

## 4. Observaciones y conclusiones

**1. El cuello de botella son las variables, no el algoritmo.** Con el mismo
preprocesamiento, los cuatro modelos caen en una franja de 0,784–0,798 de
ROC-AUC. Entre la regresión logística y XGBoost hay 0,012 puntos; entre el
peor y el mejor preprocesamiento de la regresión logística hay 0,045. Probar un
quinto algoritmo rendiría menos que mejorar los datos.

**2. La limpieza importa más donde el modelo es más rígido.** La regresión
logística gana 0,045 y los modelos de árboles entre 0,007 y 0,009. Es
coherente: los árboles absorben parte de una imputación mala o de una
codificación ordinal arbitraria haciendo cortes; un modelo lineal no puede.
Corolario práctico: si el tablero va a usar un modelo simple e interpretable, la
limpieza deja de ser opcional.

**3. El umbral por defecto de 0,5 es la peor decisión del pipeline actual.** Con
XGBoost `clean`, mover el umbral de 0,50 a 0,35 sobre las predicciones fuera de
fold cambia el resultado así:

| umbral | reservas marcadas | cancelaciones detectadas | falsas alarmas | recall | precisión | F1 |
|---|---|---|---|---|---|---|
| 0,50 | 2.910 | 2.005 | 905 | 0,558 | 0,689 | 0,616 |
| **0,35** | 4.470 | 2.675 | 1.795 | **0,744** | 0,598 | **0,663** |
| 0,25 | 5.722 | 3.028 | 2.694 | 0,842 | 0,529 | 0,650 |

El máximo de F1 se alcanza en 0,33. Bajar de 0,50 a 0,35 detecta **670
cancelaciones más** a cambio de 890 falsas alarmas adicionales. Si la acción del
negocio es barata (un correo de confirmación, una llamada, liberar el cupo para
*overbooking*) y una cancelación no detectada cuesta una noche perdida, el
intercambio es claramente favorable. El umbral debería fijarse con el costo real
de cada error, no dejarse en 0,5 por defecto. Es la decisión con más impacto de
negocio de toda la entrega y no cuesta reentrenar nada.

**4. Las variables que dominan coinciden en general con el EDA.** Por
importancia en XGBoost: `deposit_type` (No Deposit y Non Refund),
`market_segment_Online`, `has_special_requests`, `is_repeated_guest` y
`previous_cancellations` — todas señaladas en la Entrega 1. Dos matices:
`children` aparece más arriba de lo que sugería el EDA, y `lead_time` más abajo
(puesto 15), pese a haber sido una de las señales más claras del análisis
descriptivo. La explicación probable es que la importancia por ganancia se
reparte entre las columnas one-hot y penaliza a las variables continuas frente a
categóricas muy informativas; no debe leerse como un ranking causal.

**5. Limitaciones.** (a) La partición train/test es aleatoria y no temporal: el
test cubre el mismo rango de fechas que el train (2022-01 a 2025-04), así que
estas métricas **no** miden cómo envejecería el modelo con reservas futuras;
una validación temporal es el siguiente paso. (b) No se hizo búsqueda de
hiperparámetros: los valores son razonables pero no óptimos, y una búsqueda
podría cambiar el orden entre XGBoost y LightGBM. (c) El recall máximo
alcanzable ronda 0,74 con precisión aceptable; para un caso de uso que exija
más habría que incorporar variables que hoy no están en el dataset. (d) XGBoost
y random forest usan paralelismo, que introduce diferencias numéricas del orden
de 0,001 entre ejecuciones en máquinas distintas; las conclusiones no cambian,
pero las cifras exactas corresponden a los runs registrados en MLflow.

## 5. Reproducibilidad

```bash
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://ec2-32-195-106-158.compute-1.amazonaws.com:8050
python -m model.train_pipeline --modes baseline clean   # 8 runs en MLflow
python -m model.report                                  # tablas y gráfica
pytest tests/ -q                                        # 13 pruebas
```

Cada run queda en MLflow con sus parámetros, métricas por fold, métricas
agregadas, matriz de confusión, importancias de variables, predicciones fuera de
fold y el modelo serializado. Si el servidor de MLflow no responde, el pipeline
avisa y registra en un almacén local en vez de fallar.
