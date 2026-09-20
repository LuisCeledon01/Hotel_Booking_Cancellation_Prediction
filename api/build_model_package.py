"""Empaqueta el modelo entrenado como wheel para la API.

Uso, desde la raiz del repositorio:

    python api/build_model_package.py --joblib model/trained_models/xgboost__clean.joblib

Que hace:
  1. Toma el codigo del equipo que el pipeline necesita para des-serializarse
     (model/preprocessing.py y model/pipeline.py) y el archivo .joblib entrenado.
  2. Agrega la interfaz de servicio (make_prediction y el esquema de entrada).
  3. Fija las versiones EXACTAS de las librerias con que se entreno el modelo
     (un pipeline de scikit-learn solo se carga bien con la misma version).
  4. Genera api/hotel-api/model-pkg/hotel_model-<version>-py3-none-any.whl y borra
     los wheels anteriores.

Despues: reconstruir la imagen de la API (docker build) y volver a desplegarla.
Para cambiar de modelo basta con repetir estos pasos con el nuevo .joblib.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
PLANTILLA = AQUI / "model-package"
SALIDA = AQUI / "hotel-api" / "model-pkg"

# Modulos del equipo que el pickle referencia (model.pipeline, model.preprocessing)
CODIGO_EQUIPO = ["preprocessing.py", "pipeline.py"]
SIEMPRE = ["numpy", "pandas", "scikit-learn", "joblib", "scipy"]


def fijar(paquete: str) -> str:
    return f"{paquete}=={metadata.version(paquete)}"


def dependencias(joblib_path: Path) -> list[str]:
    contenido = joblib_path.read_bytes()
    deps = [fijar(p) for p in SIEMPRE]
    for lib in ("xgboost", "lightgbm"):
        if lib.encode() in contenido:
            deps.append(fijar(lib))
    deps.append("pydantic>=2.6,<3.0")
    return deps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joblib", type=Path, required=True, help="pipeline entrenado (.joblib)")
    ap.add_argument("--codigo", type=Path, default=RAIZ / "model", help="carpeta model/ del equipo")
    ap.add_argument("--nombre", help="nombre del modelo (por defecto, el del archivo)")
    ap.add_argument("--version", help="version del paquete (por defecto, la de model-package/VERSION)")
    ap.add_argument("--descripcion", default="", help="texto libre para metadata.json")
    a = ap.parse_args()

    if not a.joblib.is_file():
        sys.exit(f"No existe {a.joblib}")
    for f in CODIGO_EQUIPO:
        if not (a.codigo / f).is_file():
            sys.exit(f"Falta {a.codigo / f}")

    version = a.version or (PLANTILLA / "VERSION").read_text().strip()
    deps = dependencias(a.joblib)

    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        pkg = raiz / "model"
        shutil.copytree(PLANTILLA / "model", pkg)
        for f in CODIGO_EQUIPO:
            shutil.copy(a.codigo / f, pkg / f)
        (pkg / "trained").mkdir()
        shutil.copy(a.joblib, pkg / "trained" / "pipeline.joblib")
        (pkg / "trained" / "metadata.json").write_text(json.dumps({
            "nombre": a.nombre or a.joblib.stem,
            "version": version,
            "creado": datetime.datetime.now().isoformat(timespec="seconds"),
            "descripcion": a.descripcion,
            "python": sys.version.split()[0],
            "librerias": deps,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (pkg / "VERSION").write_text(version + "\n")
        (raiz / "setup.py").write_text(
            "from setuptools import setup\n"
            f"setup(name='hotel-model', version={version!r}, packages=['model'],\n"
            "      package_data={'model': ['VERSION', 'trained/*']}, include_package_data=True,\n"
            f"      python_requires='>=3.10', install_requires={deps!r})\n"
        )

        # pyproject.toml: pip usa su propio setuptools aislado (evita fallos con el del sistema)
        (raiz / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools>=64", "wheel"]\nbuild-backend = "setuptools.build_meta"\n'
        )

        SALIDA.mkdir(parents=True, exist_ok=True)
        for viejo in SALIDA.glob("*.whl"):
            viejo.unlink()
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(SALIDA), str(raiz)],
                       check=True)

    wheel = next(SALIDA.glob("*.whl"))
    print(f"Wheel generado: {wheel.relative_to(RAIZ) if wheel.is_relative_to(RAIZ) else wheel}")
    print("Librerias fijadas:", ", ".join(deps))
    print("Siguiente paso: docker build -t hotel-api:latest api/")


if __name__ == "__main__":
    main()
