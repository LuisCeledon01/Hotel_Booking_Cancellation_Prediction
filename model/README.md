# Pipeline de modelado

Estructura:

```
model/
  config.yml          configuracion (rutas, columnas con fuga, CV, MLflow)
  preprocessing.py    transformadores de limpieza y feature engineering
  pipeline.py         armado de los Pipeline por modelo y por configuracion
  train_pipeline.py   entrenamiento con CV y registro en MLflow
  report.py           tablas comparativas y analisis de umbral
tests/
  test_preprocessing.py
reports/
  informe_modelos.md  seccion del informe (modelos, evaluacion, conclusiones)
```

## Correr

```bash
pip install -r requirements.txt

# Servidor de MLflow (si no se define, usa el del config.yml;
# si ese no responde, cae a un almacen local sqlite)
export MLFLOW_TRACKING_URI=http://ec2-XX-XX-XX-XX.compute-1.amazonaws.com:8050

python -m model.train_pipeline                        # 4 modelos, preproc. clean
python -m model.train_pipeline --modes baseline clean # comparacion completa
python -m model.train_pipeline --models xgboost       # un solo modelo
python -m model.report                                # tablas + grafica de umbral
pytest tests/ -q
```

Los datos se esperan en `data/train.csv` y `data/test.csv` (`dvc pull`).

## Notas

- `cancellation_fee_charged` esta excluida por fuga de informacion; hay una
  prueba que falla si vuelve a entrar.
- Todo el preprocesamiento va dentro del `Pipeline`, de modo que cada fold de
  la validacion cruzada ajusta sus propios estadisticos.
- Si la instancia de EC2 con MLflow esta apagada, el entrenamiento no falla:
  avisa y registra local. Los runs locales se pueden re-registrar despues
  apuntando `MLFLOW_TRACKING_URI` al servidor y volviendo a correr.
