import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def reserva() -> dict:
    return {
        "booking_date": "2023-05-14", "arrival_date": "2023-06-24", "lead_time": 41,
        "stays_weekend": 4, "stays_weekday": 1, "adults": 2, "children": None, "babies": 0,
        "meal_plan": "Bed & Breakfast", "country": "POL", "market_segment": "Corporate",
        "room_type": "Standard", "deposit_type": "No Deposit", "agent_id": None,
        "is_repeated_guest": 1, "previous_cancellations": None,
        "previous_bookings_not_canceled": 4, "special_requests": 1,
        "avg_daily_rate": 60.37, "parking_spaces": 0,
    }
