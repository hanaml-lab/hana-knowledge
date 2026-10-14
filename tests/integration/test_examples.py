"""The examples are documentation; they must keep running."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"


@pytest.mark.parametrize("name", ["end_to_end.py", "budgeted_memory.py"])
def test_example_runs(name, capsys):
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    out = capsys.readouterr().out
    assert "Traceback" not in out and out.strip()
