# Contrato entre el tablero y la API de predicción

El tablero (`dashboard/`) no carga el modelo: lo consume por HTTP. Este documento fija
lo que el tablero envía y espera. Sigue el mismo esquema de la API del taller de Docker.

La dirección se configura con las variables de entorno **`API_URL`** (IP o nombre, sin
`http://`) y **`API_PORT`** (por defecto 8001).

## `GET /api/v1/health`

```json
{"name": "HotelRisk API", "api_version": "0.0.1", "model_version": "0.0.1"}
```

## `POST /api/v1/predict`

Solicitud: una lista de reservas en `inputs`.

```json
{
  "inputs": [
    {
      "booking_date": "2023-05-14", "arrival_date": "2023-06-24", "lead_time": 41,
      "stays_weekend": 4, "stays_weekday": 1, "adults": 2, "children": 0.0, "babies": 0,
      "meal_plan": "Bed & Breakfast", "country": "POL", "market_segment": "Corporate",
      "room_type": "Standard", "deposit_type": "No Deposit", "agent_id": 28.0,
      "is_repeated_guest": 1, "previous_cancellations": 1.0,
      "previous_bookings_not_canceled": 4, "special_requests": 1,
      "avg_daily_rate": 60.37, "parking_spaces": 0
    }
  ]
}
```

Respuesta correcta (HTTP 200):

```json
{
  "errors": null,
  "version": "0.0.1",
  "predictions": [0],
  "probabilities": [0.2874]
}
```

- **`probabilities`** es lo que usa el tablero: probabilidad de cancelación (0 a 1), en el
  mismo orden que `inputs`. Es **obligatorio** para que el umbral ajustable funcione.
  El tablero aplica el umbral sobre este valor.
- `predictions` es la clase con el umbral por defecto del modelo (0 o 1). Si la API no
  devuelve `probabilities`, el tablero usa `predictions` y avisa de que el umbral no tiene efecto.
- Ante datos inválidos la API responde HTTP 400 con `{"detail": [ ... ]}` y el tablero
  muestra el detalle al usuario.
- El tablero envía hasta 500 reservas por solicitud.

## Campos de cada reserva

Los mismos de `HotelDataInputSchema` (`package-src/model/processing/validation.py`, rama `dev`).
El tablero **no** envía `cancellation_fee_charged` (fuga de información) ni `is_canceled`.

| Campo | Tipo | Obligatorio |
|---|---|---|
| `booking_date`, `arrival_date` | texto `AAAA-MM-DD` | sí |
| `lead_time`, `stays_weekend`, `stays_weekday`, `adults`, `is_repeated_guest`, `special_requests`, `parking_spaces` | entero | sí |
| `avg_daily_rate` | decimal | sí |
| `meal_plan`, `market_segment`, `room_type`, `deposit_type` | texto | sí |
| `country` | texto | no |
| `children`, `babies`, `agent_id`, `previous_cancellations`, `previous_bookings_not_canceled` | número (puede ser `null`) | no |
| `booking_id` | entero | no |

Valores válidos de las categóricas: `room_type` Standard / Deluxe / Suite; `deposit_type`
No Deposit / Non Refund / Refundable; `market_segment` Online / Corporate / Offline /
Groups / Aviation; `meal_plan` Bed & Breakfast / Half Board / Full Board / Self Catering.
