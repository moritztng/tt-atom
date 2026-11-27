"""An unpinned open on a multi-card host pins one chip; a lone p300 chip gets the 1x1 descriptor."""
import pytest

from tt_atom import device as D


@pytest.fixture
def host(monkeypatch):
    def make(nodes, p300=()):
        monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("TT_MESH_GRAPH_DESC_PATH", raising=False)
        monkeypatch.setattr(D, "_tt_nodes", lambda: list(nodes))
        monkeypatch.setattr(D, "_is_p300", lambda n: n in p300)
        monkeypatch.setattr(D, "_mesh_descriptor", lambda f: "/mgd/" + f)
    return make


def test_multi_card_unpinned_pins_requested_chip(host, monkeypatch):
    host([0, 1, 2, 3])
    assert D.pin_one_chip(2) == 0
    import os
    assert os.environ["TT_VISIBLE_DEVICES"] == "2"
    assert "TT_MESH_GRAPH_DESC_PATH" not in os.environ


def test_lone_p300_chip_gets_descriptor(host):
    import os
    host([0, 1, 2, 3], p300=(0, 1, 2, 3))
    assert D.pin_one_chip(1) == 0
    assert os.environ["TT_MESH_GRAPH_DESC_PATH"].endswith("p150_mesh_graph_descriptor.textproto")


def test_caller_pin_and_single_card_untouched(host, monkeypatch):
    import os
    host([0])
    assert D.pin_one_chip(0) == 0
    assert "TT_VISIBLE_DEVICES" not in os.environ
    host([0, 1, 2, 3], p300=(0, 1, 2, 3))
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0,1")
    assert D.pin_one_chip(1) == 1
    assert os.environ["TT_VISIBLE_DEVICES"] == "0,1"
    assert "TT_MESH_GRAPH_DESC_PATH" not in os.environ


def test_missing_chip_refused(host):
    host([0, 1])
    with pytest.raises(ValueError):
        D.pin_one_chip(5)
