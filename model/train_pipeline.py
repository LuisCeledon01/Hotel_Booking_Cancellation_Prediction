"""Entrenamiento con validacion cruzada y registro en MLflow.

Uso:
    python -m model.train_pipeline                      # todos los modelos, modo clean
    python -m model.train_pipeline --models xgboost     # un modelo
    python -m model.train_pipeline --modes baseline clean   # comparacion

El servidor de MLflow se toma de la variable de entorno MLFLOW_TRACKING_URI o
del config.yml. Si no responde, se cae a un almacen local ./mlruns para no
bloquear el trabajo cuando la instancia de EC2 esta apagada; los runs locales
se pueden re-registrar despues apuntando al servidor.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.pipeline import build_pipeline  # noqa: E402

ALL_MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


def load_config(path: Path | None = None) -> dict:
    path = path or Path(__file__).with_name("config.yml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_tracking_uri(config: dict, timeout: float = 4.0) -> tuple[str, bool]:
    """Devuelve (uri, es_remoto). Cae al almacen local si el servidor no responde."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or config["mlflow"]["tracking_uri"]
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        return uri, False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return uri, True
    except OSError:
        print(
            f"[aviso] El servidor MLflow {uri} no responde. "
            f"Se registran los runs localmente en {config['mlflow']['fallback_uri']}.",
            file=sys.stderr,
        )
        return config["mlflow"]["fallback_uri"], False


def best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Umbral que maximiza F1 sobre las predicciones fuera de fold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[idx]), float(f1[idx])


def fold_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_proba),
    }


def run_experiment(model_name: str, mode: str, config: dict, X, y, mlflow, out_dir: Path) -> dict:
    cv_cfg = config["cv"]
    skf = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"],
    )

    oof = np.zeros(len(y), dtype=float)
    per_fold: list[dict] = []

    run_name = f"{model_name}__{mode}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model": model_name,
                "preprocessing": mode,
                "github_user": os.environ.get("GITHUB_USER", "ManoLord9408"),
                "entrega": "2",
                "leakage_excluded": ",".join(config["leakage_columns"]),
            }
        )

        pipe = build_pipeline(model_name, config, mode)
        mlflow.log_params(
            {
                "model": model_name,
                "preprocessing_mode": mode,
                "n_splits": cv_cfg["n_splits"],
                "cv_random_state": cv_cfg["random_state"],
                "rare_threshold": config["rare_threshold"],
                "adr_clip_quantile": config["adr_clip_quantile"],
            }
        )
        estimator_params = pipe.named_steps["model"].get_params()
        mlflow.log_params(
            {f"model__{k}": v for k, v in estimator_params.items() if _loggable(v)}
        )

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_va)[:, 1]
            oof[va_idx] = proba

            m = fold_metrics(y_va, proba)
            per_fold.append(m)
            for key, value in m.items():
                mlflow.log_metric(f"fold_{key}", value, step=fold)
            print(
                f"  Fold {fold} | ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} "
                f"| F1 {m['f1']:.4f} | Recall {m['recall']:.4f}"
            )

        summary = {}
        for key in per_fold[0]:
            values = [f[key] for f in per_fold]
            summary[f"cv_{key}_mean"] = float(np.mean(values))
            summary[f"cv_{key}_std"] = float(np.std(values))

        thr, thr_f1 = best_threshold(y.values, oof)
        oof_at_thr = fold_metrics(y.values, oof, thr)
        summary.update(
            {
                "oof_roc_auc": roc_auc_score(y, oof),
                "oof_pr_auc": average_precision_score(y, oof),
                "best_threshold": thr,
                "f1_at_best_threshold": thr_f1,
                "recall_at_best_threshold": oof_at_thr["recall"],
                "precision_at_best_threshold": oof_at_thr["precision"],
            }
        )
        mlflow.log_metrics(summary)

        # Artefactos
        out_dir.mkdir(parents=True, exist_ok=True)
        oof_path = out_dir / f"oof_{run_name}.csv"
        pd.DataFrame({"y_true": y.values, "y_proba": oof}).to_csv(oof_path, index=False)
        mlflow.log_artifact(str(oof_path), artifact_path="oof")

        cm = confusion_matrix(y, (oof >= thr).astype(int))
        cm_path = out_dir / f"confusion_{run_name}.json"
        cm_path.write_text(json.dumps({"threshold": thr, "matrix": cm.tolist()}, indent=2))
        mlflow.log_artifact(str(cm_path), artifact_path="diagnostics")

        # Modelo final entrenado con todos los datos.
        # Se registra con log_model (no con log_artifact) para que MLflow lo
        # reconozca como modelo: guarda la firma de entrada/salida y las
        # dependencias, aparece en la columna "Models" de la interfaz y queda
        # listo para el Model Registry y para servirlo como API.
        pipe.fit(X, y)
        model_path = out_dir / f"{run_name}.joblib"
        joblib.dump(pipe, model_path)  # copia local para el tablero
        _log_model(mlflow, pipe, X.head(5))

        importances = _feature_importances(pipe)
        if importances is not None:
            imp_path = out_dir / f"importances_{run_name}.csv"
            importances.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path), artifact_path="diagnostics")

        print(
            f"  --> ROC-AUC {summary['cv_roc_auc_mean']:.4f} (+/- {summary['cv_roc_auc_std']:.4f})"
            f" | PR-AUC {summary['cv_pr_auc_mean']:.4f}"
            f" | F1@0.5 {summary['cv_f1_mean']:.4f}"
            f" | F1@{thr:.2f} {thr_f1:.4f}"
            f" | run_id {run.info.run_id}"
        )

    return {"model": model_name, "mode": mode, **summary}



def _log_model(mlflow, pipe, input_example) -> None:
    """Registra el pipeline como modelo de MLflow.

    El nombre del parametro cambio entre MLflow 2.x (artifact_path) y 3.x
    (name), asi que se intentan los dos. Si el registro falla (por ejemplo
    porque el servidor no acepta artefactos), se avisa y se sigue: perder el
    modelo no debe tumbar el experimento.
    """
    import mlflow.sklearn

    # cloudpickle en vez del formato por defecto: el pipeline incluye
    # transformadores propios (model/preprocessing.py) que el serializador
    # nuevo de MLflow rechaza por venir de codigo del proyecto.
    kwargs = {
        "input_example": input_example,
        "serialization_format": mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    }
    try:
        try:
            mlflow.sklearn.log_model(pipe, name="model", **kwargs)
        except TypeError:
            mlflow.sklearn.log_model(pipe, artifact_path="model", **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"  [aviso] No se pudo registrar el modelo en MLflow: {exc}", file=sys.stderr)


def _loggable(value) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def _feature_importances(pipe) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        return None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        return None
    if len(values) != len(names):
        return None
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento con CV y MLflow")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--modes", nargs="+", default=["clean"], choices=["baseline", "clean"])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    import mlflow

    uri, remote = resolve_tracking_uri(config)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    print(f"MLflow -> {uri} ({'servidor remoto' if remote else 'almacen local'})")

    df = pd.read_csv(ROOT / config["data"]["train_path"])
    target = config["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]
    print(f"Datos: {X.shape[0]} filas, {X.shape[1]} columnas | tasa de cancelacion {y.mean():.4f}")

    out_dir = ROOT / config["output_dir"]
    results = []
    for mode in args.modes:
        for model_name in args.models:
            print(f"\n=== {model_name} | preprocesamiento: {mode} ===")
            results.append(
                run_experiment(model_name, mode, config, X, y, mlflow, out_dir)
            )

    table = pd.DataFrame(results)
    summary_path = out_dir / "resultados_cv.csv"
    table.to_csv(summary_path, index=False)
    cols = [
        "model",
        "mode",
        "cv_roc_auc_mean",
        "cv_pr_auc_mean",
        "cv_f1_mean",
        "cv_recall_mean",
        "best_threshold",
        "f1_at_best_threshold",
        "recall_at_best_threshold",
    ]
    print("\n=== Resumen ===")
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nTabla guardada en {summary_path}")


if __name__ == "__main__":
    main()
