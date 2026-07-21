"""Per-module parity: SO2Convolution on TT vs the fairchem golden (PCC >= 0.98)."""
import ttnn

from tt_atom.so2 import SO2Convolution
from util import pcc


def _run(golden, device, prefix, act_prefix, Cin, H, extra):
    cfg = golden.config
    w = golden.w()
    conv = SO2Convolution(
        w, prefix, device,
        sphere_channels_in=Cin, m_output_channels=H,
        lmax=cfg["lmax"], mmax=cfg["mmax"], extra_m0_output_channels=extra,
    )
    x = ttnn.from_torch(golden.act(f"{act_prefix}.in0"), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=device)
    xe = None
    if conv.has_radial:
        xe = ttnn.from_torch(golden.act(f"{act_prefix}.in1"), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device)
    out = conv(x, xe)
    if extra:
        out, extra_t = out
        e = ttnn.to_torch(extra_t).float()
        assert pcc(e, golden.act(f"{act_prefix}.out1")) >= 0.98
    o = ttnn.to_torch(out).float()
    return pcc(o, golden.act(f"{act_prefix}.out0"))


def test_so2_conv_1(golden, device):
    # Edgewise feeds 2*sphere_channels in; extra_m0 = lmax*hidden gating channels.
    cfg = golden.config
    Cin = 2 * cfg["sphere_channels"]
    extra = cfg["lmax"] * cfg["hidden_channels"]
    p = _run(golden, device, "blocks.0.edge_wise.so2_conv_1",
             "block0.so2_1", Cin, cfg["hidden_channels"], extra)
    assert p >= 0.98, f"so2_conv_1 PCC {p}"


def test_so2_conv_2(golden, device):
    cfg = golden.config
    p = _run(golden, device, "blocks.0.edge_wise.so2_conv_2",
             "block0.so2_2", cfg["hidden_channels"], cfg["sphere_channels"], 0)
    assert p >= 0.98, f"so2_conv_2 PCC {p}"
