import numpy as np
import pandas as pd
from model.predict import make_prediction

def test_make_prediction_shape_and_types(sample_input_data: pd.DataFrame):
    """Tests that predictions return expected keys, data types, and row count."""
    expected_sample_count = len(sample_input_data)

    results = make_prediction(input_data=sample_input_data)

    predictions = results.get("predictions")
    probabilities = results.get("probabilities")
    errors = results.get("errors")

    assert errors is None
    assert isinstance(predictions, np.ndarray)
    assert isinstance(probabilities, np.ndarray)
    assert len(predictions) == expected_sample_count
    assert len(probabilities) == expected_sample_count

def test_make_prediction_value_ranges(sample_input_data: pd.DataFrame):
    """Tests that probabilities lie in [0, 1] and predictions are binary (0 or 1)."""
    results = make_prediction(input_data=sample_input_data)

    predictions = results.get("predictions")
    probabilities = results.get("probabilities")

    # Check valid binary target classes
    assert set(np.unique(predictions)).issubset({0, 1})

    # Check valid probability bounds
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

def test_make_prediction_handles_data_leakage_columns(sample_input_data: pd.DataFrame):
    """Verifies that payload containing leakage columns (cancellation_fee_charged) is cleaned without failing."""
    sample_with_leakage = sample_input_data.copy()
    sample_with_leakage["cancellation_fee_charged"] = 1

    results = make_prediction(input_data=sample_with_leakage)

    assert results.get("errors") is None
    assert len(results.get("predictions")) == len(sample_input_data)

def test_make_prediction_handles_structural_missing_values(sample_input_data: pd.DataFrame):
    """Verifies that missing values in children, previous_cancellations, or country are handled gracefully."""
    sample_with_nas = sample_input_data.copy()
    sample_with_nas.loc[0, "children"] = np.nan
    sample_with_nas.loc[0, "previous_cancellations"] = np.nan
    sample_with_nas.loc[0, "country"] = np.nan

    results = make_prediction(input_data=sample_with_nas)

    assert results.get("errors") is None
    assert len(results.get("predictions")) == len(sample_input_data)