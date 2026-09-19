from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict
from model.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any] = None
    version: str
    predictions: Optional[List[int]] = None      # clase con umbral 0,5
    probabilities: Optional[List[float]] = None  # probabilidad de cancelacion (0 a 1)


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "inputs": [{
                "booking_date": "2023-05-14", "arrival_date": "2023-06-24", "lead_time": 41,
                "stays_weekend": 4, "stays_weekday": 1, "adults": 2, "children": 0.0, "babies": 0,
                "meal_plan": "Bed & Breakfast", "country": "POL", "market_segment": "Corporate",
                "room_type": "Standard", "deposit_type": "No Deposit", "agent_id": 28.0,
                "is_repeated_guest": 1, "previous_cancellations": 1.0,
                "previous_bookings_not_canceled": 4, "special_requests": 1,
                "avg_daily_rate": 60.37, "parking_spaces": 0,
            }]
        }
    })
