# Reporte de trabajo en equipo — Entrega 2

> **Borrador.** Los campos marcados con ⚠️ hay que confirmarlos con cada
> compañero antes de entregar: solo tengo evidencia directa de mi parte y de lo
> que quedó registrado en el repositorio y en el grupo de trabajo.

## Integrantes y responsabilidad principal

| Integrante | GitHub | Responsabilidad en esta entrega |
|---|---|---|
| Luis Celedón | `LuisCeledon01` | Infraestructura (EC2, servidor MLflow, DVC sobre S3) y primeros modelos |
| José M. Chadid | `jochad-dev` | Empaquetado del pipeline y redacción del informe |
| Andrés Ovalle | `Andresovalleromero-ui` | Tablero y video de funcionamiento |
| Manuel Linares | `ManoLord9408` | Preprocesamiento, comparación de modelos e integración con MLflow |

## Actividades realizadas y evidencia

**Luis Celedón — infraestructura y modelos iniciales.** Montó la instancia EC2
con el servidor de MLflow en el puerto 8050, configuró el versionado de datos
con DVC contra un bucket S3 y desarrolló los dos primeros modelos
(`random_forest.ipynb` y `xgboost.ipynb`), con validación cruzada de 5
particiones y registro de experimentos en MLflow. Fue quien habilitó que el
resto pudiéramos trabajar sobre una misma fuente de datos y un mismo servidor
de seguimiento.
*Evidencia:* commits en el repositorio, notebooks en `notebooks/`, runs
`RandomForest_1..6` y `XGBoost_1..4` en MLflow.

**José M. Chadid — empaquetado y documentación.** Tomó como base la estructura
del laboratorio 5 y la adaptó al proyecto, dejando el pipeline empaquetado con
`tox` y pruebas automatizadas con `pytest`, de modo que el entrenamiento se
ejecuta con un comando y queda listo para exponerse como API. Además redactó la
versión consolidada del informe (contexto, problema, alcance, modelado y
conclusiones).
*Evidencia:* ⚠️ repositorio del paquete y documento v3 en el Drive compartido.

**Andrés Ovalle — tablero.** Desarrolló la interfaz del tablero siguiendo la
pregunta de negocio y el alcance definidos en la Entrega 1, e hizo el video que
evidencia su funcionamiento.
*Evidencia:* ⚠️ commits del tablero y enlace al video.

**Manuel Linares — preprocesamiento y evaluación comparativa.** A partir de los
hallazgos del EDA de la Entrega 1, desarrolló el módulo de limpieza y selección
de características: exclusión de `cancellation_fee_charged` por fuga de
información, imputación por dominio de `previous_cancellations` y `children`,
derivación de variables de calendario, agrupación de categorías de baja
frecuencia y tratamiento de atípicos. Integró el entrenamiento con el servidor
de MLflow y montó una comparación pareada de cuatro modelos bajo dos
configuraciones de preprocesamiento, más el análisis de umbral de decisión.
*Evidencia:* rama `manuel/preprocesamiento-y-mlflow`, 13 pruebas automatizadas,
8 runs registrados en MLflow, `reports/informe_modelos.md`.

## Cómo se distribuyó y coordinó el trabajo

El reparto se hizo por componente del producto de datos —infraestructura,
modelos, empaquetado, tablero— para poder avanzar en paralelo. La coordinación
fue por el grupo de mensajería, con una reunión de sincronización el 5 de
septiembre para unificar avances y evitar duplicidad de trabajo. El código se
integró mediante ramas y *pull requests* sobre el repositorio del proyecto.

## Dificultades encontradas

1. **Disponibilidad de la infraestructura.** La instancia EC2 corre sobre AWS
   Academy, cuyas sesiones expiran; estuvo apagada en varios momentos y el DNS
   público cambia al reiniciarla. Se mitigó dejando que el pipeline de
   entrenamiento detecte si el servidor no responde y registre localmente, para
   no bloquear el trabajo, y re-registrando después contra el servidor.
2. **Trabajo duplicado al inicio.** Dos integrantes empezaron a montar el
   pipeline de modelos por separado y se crearon dos repositorios. Se detectó en
   la reunión del 5 de septiembre y se resolvió unificando en un solo
   repositorio y repartiendo explícitamente los frentes.
3. **Distribución desigual de la carga.** ⚠️ Uno de los integrantes se
   incorporó tarde por compromisos laborales y asumió un frente adicional
   (preprocesamiento, comparación e integración con MLflow) para compensar.

## Reparto previsto para la Entrega 3

| Integrante | Siguiente frente |
|---|---|
| Luis | Despliegue y disponibilidad del servicio |
| José | API de predicción sobre el paquete existente |
| Andrés | Conexión del tablero con la API |
| Manuel | Búsqueda de hiperparámetros y validación temporal |

⚠️ Confirmar este reparto con el equipo antes de entregar.
