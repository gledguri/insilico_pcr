import pytest

from ispcr import config, paths


@pytest.fixture(autouse=True)
def _keep_data_folder():
    """Tests may switch the data folder; put the real one back afterwards."""
    old = paths.ROOT
    yield
    paths.ROOT = old
    config.settings.cache_clear()
