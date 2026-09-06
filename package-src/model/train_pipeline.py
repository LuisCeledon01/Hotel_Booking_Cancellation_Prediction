import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from model.config.core import config, PACKAGE_ROOT
from model.pipeline import get_pipeline
from model.processing.validation import validate_inputs


def get_model_instance(model_name: str, params: dict):
    """Factory function for selecting and initializing models."""
    models = {
        "lightgbm": LGBMClassifier(**params.get("lightgbm", {})),
        "xgboost": XGBClassifier(**params.get("xgboost", {})),
        "logistic_regression": LogisticRegression(**params.get("logistic_regression", {}))
    }
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(models.keys())}")
    return models[model_name]


def evaluate_predictions(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculates evaluation metrics for binary classification."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "ROC-AUC": round(auc_roc, 4),
        "PR-AUC": round(auc_pr, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Confusion_Matrix": cm
    }


def run_training_with_cv():
    """Runs model training with Stratified K-Fold Cross-Validation."""
    data_path = PACKAGE_ROOT / "datasets" / "train.csv"
    raw_df = pd.read_csv(data_path)

    # Validate inputs & drop leakage
    validated_df, _ = validate_inputs(input_data=raw_df)
    
    X = validated_df.drop(columns=[config.model_settings.target])
    y = raw_df[config.model_settings.target].values

    selected_model_name = config.model_settings.selected_model
    model_instance = get_model_instance(selected_model_name, config.model_settings.model_params)
    pipeline = get_pipeline(model_instance, config.model_settings)

    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.model_settings.random_state)
    
    cv_scores = {"ROC-AUC": [], "PR-AUC": [], "F1-Score": [], "Recall": []}

    print(f"\n==================================================")
    print(f" Starting 5-Fold Stratified CV for: [{selected_model_name.upper()}] ")
    print(f"==================================================")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        pipeline.fit(X_train, y_train)
        preds_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics = evaluate_predictions(y_val, preds_proba)
        
        cv_scores["ROC-AUC"].append(metrics["ROC-AUC"])
        cv_scores["PR-AUC"].append(metrics["PR-AUC"])
        cv_scores["F1-Score"].append(metrics["F1-Score"])
        cv_scores["Recall"].append(metrics["Recall"])

        print(f"Fold {fold} | ROC-AUC: {metrics['ROC-AUC']} | PR-AUC: {metrics['PR-AUC']} | Recall: {metrics['Recall']} | F1: {metrics['F1-Score']}")

    print(f"\n--- Mean CV Performance [{selected_model_name}] ---")
    for metric_name, values in cv_scores.items():
        print(f"{metric_name}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    # Fit pipeline on full dataset and save artifact
    pipeline.fit(X, y)

    save_path = PACKAGE_ROOT / "trained_models" / f"hotel_cancellation_{selected_model_name}.joblib"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    
    print(f"\nModel artifact saved successfully at: {save_path}")


if __name__ == "__main__":
    run_training_with_cv()