"""import tt_atom turns numpy's huge-page hint off unless the user set it."""
import os
import subprocess
import sys

PROBE = "import numpy, tt_atom, os; m = numpy.core.multiarray if numpy.__version__ < '2' else numpy._core.multiarray; print(os.environ['NUMPY_MADVISE_HUGEPAGE'], int(m._set_madvise_hugepage(False)))"


def _run(value=None):
    env = {k: v for k, v in os.environ.items() if k != "NUMPY_MADVISE_HUGEPAGE"}
    if value is not None:
        env["NUMPY_MADVISE_HUGEPAGE"] = value
    return subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True).stdout.split()


def test_hint_off_when_numpy_was_imported_first():
    assert _run() == ["0", "0"]          # _set_madvise_hugepage returns the previous setting


def test_user_setting_wins():
    assert _run("1") == ["1", "1"]
