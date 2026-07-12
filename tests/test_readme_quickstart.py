"""DONE_CHECK for the unified public entry point: the new `Calculator` exists AND the README's
own Quickstart demonstrates it as a runnable example spanning both model families. Host-only (no
device): it proves the entry point is importable, that the Quickstart shows it in real code, that
that code is valid Python, and that both a UMA and an Orb checkpoint are demonstrated. On-device
bit-exact parity through `Calculator` is covered by `tests/test_orb_calculator.py`."""
import ast
import pathlib
import re

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _quickstart():
    txt = README.read_text()
    return txt.split("## Quickstart", 1)[1].split("\n## ", 1)[0]


def test_unified_entry_point_exists():
    import tt_atom

    assert callable(tt_atom.Calculator)
    from tt_atom import Calculator  # the documented import must work
    assert Calculator is tt_atom.Calculator


def test_quickstart_demonstrates_calculator():
    qs = _quickstart()
    assert "from tt_atom import Calculator" in qs, "Quickstart must import the unified entry point"
    blocks = re.findall(r"```python\n(.*?)```", qs, re.S)
    assert blocks, "Quickstart must contain a python example"
    code = "\n".join(blocks)
    assert "Calculator(" in code, "Quickstart must call Calculator(...)"
    for b in blocks:
        ast.parse(b)  # every shown example must be valid Python


def test_quickstart_covers_both_families():
    qs = _quickstart()
    assert "uma-s-1" in qs, "Quickstart must demonstrate a UMA checkpoint"
    assert re.search(r"orb-v3-\w", qs), "Quickstart must demonstrate an Orb checkpoint"
