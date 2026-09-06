"""Genera la comparacion de experimentos y el analisis de umbral.

Lee los runs registrados en MLflow (no vuelve a entrenar) y produce:
  - reports/comparacion_modelos.csv  : tabla baseline vs clean por modelo
  - reports/prueba_pareada.csv       : diferencia por fold + t-test pareado
  - reports/umbral_<modelo>.csv      : barrido de umbral del mejor modelo
  - reports/umbral.png               : curva precision/recall/F1 vs umbral

Uso:
    python -m model.report
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.train_pipeline import load_config, resolve_tracking_uri  # noqa: E402

MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


def _fold_history(client, run_id: str, metric: str) -> list[float]:
    return [m.value for m in client.get_metric_history(run_id, metric)]


def main() -> None:
    import mlflow

    config = load_config()
    uri, _ = resolve_tracking_uri(config)
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if exp is None:
        raise SystemExit("No hay experimento en MLflow. Corra primero train_pipeline.")

    runs = client.search_runs([exp.experiment_id], max_results=500)
    latest: dict[str, object] = {}
    for r in sorted(runs, key=lambda r: r.info.start_time):
        name = r.data.tags.get("mlflow.runName")
        if name:
            latest[name] = r  # se queda con el mas reciente

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)

    # --- Tabla comparativa -------------------------------------------------
    filas = []
    for name, run in latest.items():
        if "__" not in name:
            continue
        model, mode = name.split("__", 1)
        m = run.data.metrics
        filas.append(
            {
                "modelo": model,
                "preprocesamiento": mode,
                "roc_auc": m.get("cv_roc_auc_mean"),
                "roc_auc_std": m.get("cv_roc_auc_std"),
                "pr_auc": m.get("cv_pr_auc_mean"),
                "f1_umbral_0.5": m.get("cv_f1_mean"),
                "recall_umbral_0.5": m.get("cv_recall_mean"),
                "umbral_optimo": m.get("best_threshold"),
                "f1_umbral_optimo": m.get("f1_at_best_threshold"),
                "recall_umbral_optimo": m.get("recall_at_best_threshold"),
                "run_id": run.info.run_id,
            }
        )
    tabla = pd.DataFrame(filas).sort_values(["preprocesamiento", "roc_auc"], ascending=[True, False])
    tabla.to_csv(out / "comparacion_modelos.csv", index=False)
    print("=== Comparacion de modelos ===")
    print(tabla.drop(columns=["run_id"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- Prueba pareada por fold ------------------------------------------
    from scipy import stats

    pareado = []
    for model in MODELS:
        b, c = latest.get(f"{model}__baseline"), latest.get(f"{model}__clean")
        if not (b and c):
            continue
        vb = np.array(_fold_history(client, b.info.run_id, "fold_roc_auc"))
        vc = np.array(_fold_history(client, c.info.run_id, "fold_roc_auc"))
        if len(vb) != len(vc) or len(vb) < 3:
            continue
        t, p = stats.ttest_rel(vc, vb)
        pareado.append(
            {
                "modelo": model,
                "roc_auc_baseline": vb.mean(),
                "roc_auc_clean": vc.mean(),
                "diferencia": (vc - vb).mean(),
                "desv_diferencia": (vc - vb).std(ddof=1),
                "p_valor_pareado": p,
            }
        )
    if pareado:
        dfp = pd.DataFrame(pareado)
        dfp.to_csv(out / "prueba_pareada.csv", index=False)
        print("\n=== Limpieza vs baseline (mismos folds, t pareado) ===")
        print(dfp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(
            "\nNota: 5 folds son pocos y comparten datos entre si, asi que el t "
            "pareado sobre folds es optimista (Dietterich, 1998). Sirve para "
            "descartar que la mejora sea puro ruido, no como prueba formal."
        )

    # --- Barrido de umbral del mejor modelo --------------------------------
    mejor = tabla[tabla.preprocesamiento == "clean"].iloc[0]
    oof_path = ROOT / config["output_dir"] / f"oof_{mejor.modelo}__clean.csv"
    if not oof_path.exists():
        print(f"\n[aviso] No se encontro {oof_path}; se omite el barrido de umbral.")
        return

    oof = pd.read_csv(oof_path)
    y, proba = oof["y_true"].values, oof["y_proba"].values
    filas = []
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        filas.append(
            {
                "umbral": round(float(thr), 2),
                "reservas_marcadas": int(pred.sum()),
                "cancelaciones_detectadas": tp,
                "falsas_alarmas": fp,
                "cancelaciones_perdidas": fn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
    barrido = pd.DataFrame(filas)
    barrido.to_csv(out / f"umbral_{mejor.modelo}.csv", index=False)
    print(f"\n=== Barrido de umbral | {mejor.modelo} (clean) ===")
    print(barrido.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(barrido.umbral, barrido.precision, marker="o", label="Precision")
    ax.plot(barrido.umbral, barrido.recall, marker="s", label="Recall")
    ax.plot(barrido.umbral, barrido.f1, marker="^", label="F1")
    ax.axvline(0.5, ls="--", lw=1, color="grey")
    ax.annotate("umbral por defecto", xy=(0.5, 0.05), xytext=(0.53, 0.05), fontsize=8, color="grey")
    ax.axvline(float(mejor.umbral_optimo), ls="--", lw=1, color="crimson")
    ax.annotate(
        f"optimo F1 = {float(mejor.umbral_optimo):.2f}",
        xy=(float(mejor.umbral_optimo), 0.95),
        xytext=(float(mejor.umbral_optimo) + 0.03, 0.95),
        fontsize=8,
        color="crimson",
    )
    ax.set_xlabel("Umbral de decision")
    ax.set_ylabel("Metrica")
    ax.set_title(f"Precision, recall y F1 segun el umbral - {mejor.modelo}")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "umbral.png", dpi=150)
    print(f"\nArchivos generados en {out}")


if __name__ == "__main__":
    main()
