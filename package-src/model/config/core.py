from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel
import yaml

# Path setup
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """Application-level configuration."""

    package_name: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """Model training & feature engineering configuration."""

    target: str
    drop_features: List[str]
    date_features: List[str]
    numerical_features: List[str]
    categorical_features: Sequence[str]
    test_size: float
    random_state: int
    selected_model: str
    model_params: Dict[str, Dict[str, Any]]


class Config(BaseModel):
    """Master config object combining AppConfig and ModelConfig."""

    app_config: AppConfig
    model_settings: ModelConfig


def find_config_file() -> Path:
    """Locate configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config file not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> dict:
    """Parse YAML config file into native Python types."""
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as f:
            parsed_config = yaml.safe_load(f)
            return parsed_config
    raise FileNotFoundError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: Optional[dict] = None) -> Config:
    """Validate config data structures using Pydantic."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_settings=ModelConfig(**parsed_config),
    )
    return _config


config = create_and_validate_config()