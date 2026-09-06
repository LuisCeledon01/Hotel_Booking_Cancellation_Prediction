# Hotel Booking Cancellation Prediction

Microproyecto del curso **Proyecto: Desarrollo de Soluciones** (Universidad de los Andes). El objetivo es predecir la **cancelación de reservas hoteleras** a partir del historial de reservas y las características del cliente, para permitir acciones preventivas (confirmaciones, depósitos) antes de la fecha de llegada.

## Datos

El conjunto de datos proviene del reto [Hotel Booking Cancellation Prediction (Kaggle)](https://www.kaggle.com/competitions/fdd-exercise-hotel-booking-cancellation-prediction/overview): 9.696 reservas para entrenamiento y 2.424 para prueba. La variable objetivo es `is\_canceled` (1 = cancelada). Las variables incluyen aspectos temporales (fecha de reserva, `lead\_time`), de la estancia (noches, huéspedes, tipo de habitación, tarifa), del cliente (país, canal, huésped repetido) y contractuales (tipo de depósito).

Los datos se versionan con **DVC** sobre un remote en **S3**:

```bash
dvc pull
```

## Modelos y resultados

Se entrenaron y compararon dos modelos principales: **Random Forest** y **XGBoost**, con seguimiento de experimentos en **MLflow** y validación cruzada estratificada de 5 particiones.

|Métrica|Random Forest|XGBoost|
|-|-|-|
|Accuracy|72,58%|**73,97%**|
|Recall (clase 1)|43,67%|**55,77%**|
|Precisión (clase 1)|**71,20%**|68,20%|
|Especificidad (clase 1)|**89,64%**|84,68%|

**XGBoost** obtuvo el mejor desempeño global, con mayor recall en la clase de cancelación (reduciendo los falsos negativos de 405 a 318 frente a Random Forest). Dado que el objetivo de negocio es anticipar reservas en riesgo para activar acciones preventivas, se prioriza el recall de la clase 1 sin descuidar las demás métricas. **XGBoost es el modelo candidato para producción.**

## Estructura del repositorio

```
.
├── data/         # Datos crudos y procesados (versionados con DVC)
├── notebooks/    # Notebooks de análisis exploratorio y modelado
├── .dvc/         # Configuración y metadatos de DVC (remote: S3)
├── .dvcignore
└── .gitignore
```

## Tablero (HotelRisk)

Prototipo funcional en **Streamlit** con tres pantallas: Dashboard (consulta puntual del riesgo de cancelación), Analytics (tasas de cancelación por segmento) y Reports (historial de consultas en lote). Actualmente usa un modelo *placeholder*; el siguiente paso es reemplazarlo por XGBoost y conectar el tablero a una API.

## Cómo reproducir el proyecto

```bash
git clone https://github.com/LuisCeledon01/Hotel\_Booking\_Cancellation\_Prediction.git
cd Hotel\_Booking\_Cancellation\_Prediction
pip install -r requirements.txt
dvc pull
```

## Equipo

Andrés Ovalle Romero · Luis Alfonso Celedón Rocha · Jose Chadid · Manuel Alberto Linares Yepes

## 

