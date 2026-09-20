"""Paquete del modelo de cancelacion de reservas (lo instala la API).

No se edita a mano: lo arma api/build_model_package.py a partir del codigo del
equipo (model/preprocessing.py, model/pipeline.py) y del pipeline entrenado.
"""
import json
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent

with open(_RAIZ / "VERSION") as _f:
    __version__ = _f.read().strip()


def metadata() -> dict:
    """Datos del modelo empaquetado (nombre, fecha, librerias)."""
    try:
        return json.loads((_RAIZ / "trained" / "metadata.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
