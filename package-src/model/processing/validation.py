from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from model.config.core import config, PACKAGE_ROOT


class HotelDataInputSchema(BaseModel):
    """Schema validation for incoming hotel booking payloads."""
    booking_id: Optional[int] = None
    booking_date: str
    arrival_date: str
    lead_time: int
    stays_weekend: int
    stays_weekday: int
    adults: int
    children: Optional[float] = 0.0
    babies: Optional[float] = 0.0
    meal_plan: str
    country: Optional[str] = "Unknown"
    market_segment: str
    room_type: str
    deposit_type: str
    agent_id: Optional[float] = None
    is_repeated_guest: int
    previous_cancellations: Optional[float] = 0.0
    previous_bookings_not_canceled: Optional[float] = 0.0
    special_requests: int
    avg_daily_rate: float
    parking_spaces: int


class MultipleHotelDataInputs(BaseModel):
    inputs: List[HotelDataInputSchema]


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Validates input raw DataFrame:
    1. Removes leakage features (cancellation_fee_charged).
    2. Handles missing default values for structural NAs.
    3. Validates types and constraints against Pydantic schema.
    """
    validated_df = input_data.copy()
    errors = None

    # Step 1: Strip data leakage columns if present in payload
    for drop_col in config.model_settings.drop_features:
        if drop_col in validated_df.columns:
            validated_df.drop(columns=[drop_col], inplace=True)

    # Step 2: Handle structural missing values prior to schema check
    validated_df["children"] = validated_df["children"].fillna(0)
    validated_df["previous_cancellations"] = validated_df["previous_cancellations"].fillna(0)
    validated_df["country"] = validated_df["country"].fillna("Unknown")

    # Step 3: Validate rows via Pydantic schema
    try:
        records = validated_df.to_dict(orient="records")
        MultipleHotelDataInputs(inputs=records)
    except ValidationError as err:
        errors = err.errors()

    return validated_df, errors