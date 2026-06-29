"""Per-module parity: RMS norm SH on TT vs the fairchem golden (PCC >= 0.98)."""
import ttnn

from tt_atom.norm import RMSNormSH
from util import pcc


def test_rms_norm_sh(golden, device):
    cfg = golden.config
    norm = RMSNormSH(golden.w(), "blocks.0.norm_1", device,
                     lmax=cfg["lmax"], num_channels=cfg["sphere_channels"])
    x = ttnn.from_torch(golden.act("block0.norm_1.in0"), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(norm(x)).float()
    p = pcc(out, golden.act("block0.norm_1.out0"))
    assert p >= 0.98, f"rms_norm_sh PCC {p}"
