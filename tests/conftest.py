from typing import Any

import pytest

from prefix.data import load_harmbench


@pytest.fixture(scope="session")
def harmbench_records(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    return load_harmbench(tmp_path_factory.mktemp("harmbench"))
