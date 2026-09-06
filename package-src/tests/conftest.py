import pytest
import pandas as pd
from model.config.core import config, PACKAGE_ROOT

@pytest.fixture
def sample_input_data() -> pd.DataFrame:
    """Loads a slice of sample data from test.csv for pytest runs."""
    data_path = PACKAGE_ROOT / "datasets" / "test.csv"
    df = pd.read_csv(data_path)
    return df.head(10).copy()