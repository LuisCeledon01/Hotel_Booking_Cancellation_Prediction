# API de predicción HotelRisk

API FastAPI que sirve el modelo de cancelación de reservas. Sigue el molde del taller de
Docker (`bankchurn-api`): la app no contiene el modelo, lo instala como un paquete
`.whl` (`hotel-api/model-pkg/`). El contrato con el tablero está en `../dashboard/CONTRATO_API.md`.

```
api/
├── Dockerfile
├── build_model_package.py   convierte un modelo entrenado (.joblib) en el wheel
├── model-package/           interfaz de servicio que se agrega al modelo (predict, esquema)
└── hotel-api/
    ├── app/                 la API (main, api, schemas, tests)
    ├── model-pkg/           wheel del modelo (lo genera build_model_package.py)
    ├── requirements.txt
    └── run.sh
```

## Cambiar de modelo (los 3 pasos)

Desde la raíz del repositorio, con el entorno donde se entrenó el modelo:

```bash
# 1. Entrenar (genera model/trained_models/<modelo>.joblib)
python -m model.train_pipeline --models xgboost

# 2. Empaquetar: fija las versiones exactas de las librerías y genera el wheel
python api/build_model_package.py --joblib model/trained_models/xgboost__clean.joblib \
       --nombre xgboost__clean --version 0.1.0

# 3. Reconstruir y volver a desplegar la imagen
sudo docker build -t hotel-api:latest api/
```

- El wheel usa `model/preprocessing.py` y `model/pipeline.py` del repositorio: deben ser los
  mismos con los que se entrenó el `.joblib` (un pipeline solo se carga con el código que lo creó).
- Si el nuevo modelo usa **otras variables de entrada**, hay que actualizar `DataInputSchema` en
  `model-package/model/validation.py` y `dashboard/CONTRATO_API.md`.
- Si cambia el modelo, regenerar también la curva del umbral del tablero
  (`dashboard/scripts/build_assets.py --umbral reports/umbral_<modelo>.csv`).

## Probar en local

```bash
cd api/hotel-api
pip install -r test_requirements.txt
pip install model-pkg/*.whl
pytest app/tests -q
PORT=8001 bash run.sh        # http://localhost:8001/docs
```

## Contenedor

```bash
sudo docker build -t hotel-api:latest api/
sudo docker run -p 8001:8001 -it -e PORT=8001 hotel-api
```

Nota: `xgboost` necesita `libgomp1`, que la imagen `python:3.12-slim` no trae; el `Dockerfile` lo instala.
