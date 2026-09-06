import typing as t
import pandas as pd
import joblib

from model.config.core import config, PACKAGE_ROOT
from model.processing.validation import validate_inputs

# Dynamically target model based on config setting
_model_filename = f"hotel_cancellation_{config.model_settings.selected_model}.joblib"
_model_path = PACKAGE_ROOT / "trained_models" / _model_filename


def _load_pipeline():
    """Loads saved pipeline artifact from disk."""
    if not _model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {_model_path}. "
            "Please run 'python hotel_cancellation_model/train_pipeline.py' first."
        )
    return joblib.load(filename=_model_path)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict, t.List[dict]]) -> dict:
    """
    Generates prediction probabilities for booking cancellations.
    Strips data leakage (cancellation_fee_charged) and applies structural imputation.
    """
    if isinstance(input_data, dict):
        data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        data = pd.DataFrame(input_data)
    else:
        data = input_data.copy()

    # Step 1: Validate payload and handle leakage/NAs
    validated_data, errors = validate_inputs(input_data=data)

    results = {"predictions": None, "probabilities": None, "errors": errors}

    if not errors:
        pipeline = _load_pipeline()
        
        # Step 2: Inference
        probabilities = pipeline.predict_proba(validated_data)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results["predictions"] = predictions
        results["probabilities"] = probabilities

    return results