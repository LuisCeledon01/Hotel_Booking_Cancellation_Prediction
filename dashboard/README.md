# Tablero HotelRisk

Tablero en Streamlit con tres pantallas (Dashboard, Analytics, Reports). Consume la API de
predicción por HTTP (ver `CONTRATO_API.md`) y tiene un **umbral de decisión ajustable** en la
barra lateral.

```
dashboard/
├── Dockerfile
├── CONTRATO_API.md        qué envía el tablero y qué espera de la API
├── app/                   lo que entra al contenedor
│   ├── app.py             pantallas
│   ├── api_client.py      cliente de la API
│   ├── umbral.py          umbral y bandas de riesgo
│   ├── assets/            agregados pequeños (sin datos crudos ni modelo)
│   ├── requirements.txt
│   └── run.sh
├── scripts/build_assets.py  regenera assets/ desde data/train.csv y reports/umbral_xgboost.csv
└── tools/mock_api.py        API simulada para probar sin la API real
```

## Probar en local con la API simulada

```powershell
pip install -r dashboard/app/requirements.txt fastapi "uvicorn<0.30"

# Terminal 1 (desde dashboard/): API simulada en el puerto 8001
uvicorn tools.mock_api:app --port 8001

# Terminal 2 (desde dashboard/app/)
$env:API_URL="localhost"; $env:API_PORT="8001"; $env:PORT="8050"
streamlit run app.py --server.port=8050
```

Abrir http://localhost:8050. La API simulada NO usa el modelo entrenado: solo sirve para
probar la interfaz.

## Contenedor

```bash
# desde la raíz del repo
sudo docker build -t hotelrisk-dash:latest dashboard/
sudo docker run -p 8050:8050 -it -e PORT=8050 -e API_URL=X.Y.Z.W -e API_PORT=8001 hotelrisk-dash
```

`X.Y.Z.W` es la IP pública de la máquina donde corre la API (igual que en el taller 8).
