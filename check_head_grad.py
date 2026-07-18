"""Layer-by-layer FD check: perturb the device forward's INTERMEDIATE outputs (out_node, out_edge
of each layer) and compare the adjoint propagated to edge_vec_cat / cutoff / log_mask against a
single-layer FD. Isolates which layer's backward is wrong."""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
import pathlib
import numpy as np
import torch

WEIGHTS = str(pathlib.Path.home() / ".cache/tt_atom/pet_weights/pet-mad-s-v1.5.0.npz")
GOLDEN = "tests/data/pet_mad_s_si_golden.npz"


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    import ttnn
    from tt_atom.device import open_device
    from tt_atom.pet_weights import PetWeights
    from tt_atom.pet_geometry import host_pet_geometry
    from tt_atom.pet_model import PetModel, build_device_inputs
    from tt_atom.pet_vjp import _energy_head_bw, _gnn_layer_bw

    gw = PetWeights.load(WEIGHTS)
    fx = np.load(GOLDEN)
    pos = torch.tensor(fx["positions"], dtype=torch.float64)
    numbers = torch.tensor(fx["numbers"], dtype=torch.long)
    cell = torch.tensor(fx["cell"], dtype=torch.float64)
    pbc = torch.tensor(fx["pbc"], dtype=torch.bool)
    scale = gw.energy_scale()
    comp = gw.composition_energy_by_z()

    dev = open_device(0)
    try:
        model = PetModel(gw.weights, dev, cfg=gw.config)
        bd = host_pet_geometry(pos.detach().to(torch.float64), numbers, cell=cell, pbc=pbc, cfg=gw.config)
        bd_dev = build_device_inputs(bd, gw.config, dev)
        raw = model.forward(bd_dev)
        g_evc_dev, g_cut_dev, g_lm_dev = model.backward(bd_dev, g_raw=1.0)
        g_evc = ttnn.to_torch(g_evc_dev).float().numpy()
        g_cut = ttnn.to_torch(g_cut_dev).float().numpy()
        g_lm = ttnn.to_torch(g_lm_dev).float().numpy()
        print(f"g_evc sum {g_evc.sum():.4f}  g_cut sum {g_cut.sum():.4f}  g_lm sum {g_lm.sum():.4f}")

        # ---- Check energy-head backward in isolation ----
        # Re-run forward to populate caches, then perturb the head's input (node_feat, edge_feat)
        # and FD the head output (raw energy). Compare to _energy_head_bw(node=ones, cutoff).
        eh = model.energy_head
        node_feat = ttnn.to_torch(ttnn.embedding(bd_dev["node_idx"], model.node_emb_w)).float()
        # final node_emb / input_edge after the full forward -- recompute by re-running forward
        # (cheap) and grabbing the final node_emb/input_edge. Easier: just re-run forward.
        node_emb_dev = None
        input_edge = ttnn.embedding(bd_dev["elem_nbr"], model.edge_emb_w)
        input_edge = ttnn.reshape(input_edge, (bd_dev["N"], bd_dev["Dmax"], gw.config["d_pet"]))
        node_emb = ttnn.embedding(bd_dev["node_idx"], model.node_emb_w)
        for layer in model.layers:
            node_emb, input_edge, _ = (lambda on, ie, oe, nie: (ie, oe, nie))(*None, None, None) if False else (None, None, None)
        # The above is messy; just re-run the forward cleanly:
        node_emb = ttnn.embedding(bd_dev["node_idx"], model.node_emb_w)
        input_edge = ttnn.embedding(bd_dev["elem_nbr"], model.edge_emb_w)
        input_edge = ttnn.reshape(input_edge, (bd_dev["N"], bd_dev["Dmax"], gw.config["d_pet"]))
        for layer in model.layers:
            node_emb, out_edge_pre, input_edge = layer(
                node_emb, input_edge, bd_dev["edge_vec_cat"], bd_dev["elem_nbr"],
                bd_dev["log_mask"], bd_dev["rev_idx"],
                rev_idx_host=bd_dev.get("rev_idx_host"))
        # now node_emb, input_edge are the head's inputs. Perturb them for FD.
        nf0 = ttnn.to_torch(node_emb).float().numpy()
        ef0 = ttnn.to_torch(input_edge).float().numpy()
        cf0 = ttnn.to_torch(bd_dev["cutoff_factors"]).float().numpy()

        def head_energy(nf, ef, cf):
            nf_t = ttnn.from_torch(torch.tensor(nf, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev)
            ef_t = ttnn.from_torch(torch.tensor(ef, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev)
            cf_t = ttnn.from_torch(torch.tensor(cf, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev)
            r = eh(nf_t, ef_t, cf_t)
            return float(ttnn.to_torch(r).float().view(-1)[0])

        # adjoint via _energy_head_bw with g_raw=1
        g_seed = ttnn.ones((1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        g_nf, g_ef, g_cf = _energy_head_bw(eh, g_seed, bd_dev["cutoff_factors"])
        g_nf_np = ttnn.to_torch(g_nf).float().numpy()
        g_ef_np = ttnn.to_torch(g_ef).float().numpy()
        g_cf_np = ttnn.to_torch(g_cf).float().numpy()

        eps = 1.0
        # FD node_feat (sample 50 entries)
        fd_nf = np.zeros_like(nf0)
        rng = np.random.default_rng(1)
        Nn, dn = nf0.shape
        idxs = rng.choice(Nn * dn, size=min(50, Nn * dn), replace=False)
        for s in idxs:
            n, k = divmod(s, dn)
            pp = nf0.copy(); pp[n, k] += eps
            pm = nf0.copy(); pm[n, k] -= eps
            fd_nf[n, k] = (head_energy(pp, ef0, cf0) - head_energy(ef0, pm, cf0)) / (2 * eps)
        mask = np.zeros_like(fd_nf, dtype=bool)
        for s in idxs:
            n, k = divmod(s, dn); mask[n, k] = True
        print(f"[FD head node_feat] PCC={_pcc(g_nf_np[mask], fd_nf[mask]):.4f} "
              f"max|g|={np.abs(g_nf_np[mask]).max():.3f} max|fd|={np.abs(fd_nf[mask]).max():.3f}")

        # FD edge_feat (sample 50)
        fd_ef = np.zeros_like(ef0)
        Nn, Dm, dp = ef0.shape
        idxs = rng.choice(Nn * Dm * dp, size=min(50, Nn * Dm * dp), replace=False)
        for s in idxs:
            n, d, k = np.unravel_index(s, (Nn, Dm, dp))
            pp = ef0.copy(); pp[n, d, k] += eps
            pm = ef0.copy(); pm[n, d, k] -= eps
            fd_ef[n, d, k] = (head_energy(nf0, pp, cf0) - head_energy(nf0, pm, cf0)) / (2 * eps)
        mask = np.zeros_like(fd_ef, dtype=bool)
        for s in idxs:
            n, d, k = np.unravel_index(s, (Nn, Dm, dp)); mask[n, d, k] = True
        print(f"[FD head edge_feat] PCC={_pcc(g_ef_np[mask], fd_ef[mask]):.4f} "
              f"max|g|={np.abs(g_ef_np[mask]).max():.3f} max|fd|={np.abs(fd_ef[mask]).max():.3f}")

        # FD cutoff_factors (sample 50)
        fd_cf = np.zeros_like(cf0)
        Nn, Dm, _ = cf0.shape
        idxs = rng.choice(Nn * Dm, size=min(50, Nn * Dm), replace=False)
        for s in idxs:
            n, d = divmod(s, Dm)
            pp = cf0.copy(); pp[n, d, 0] += eps
            pm = cf0.copy(); pm[n, d, 0] -= eps
            fd_cf[n, d, 0] = (head_energy(nf0, ef0, pp) - head_energy(nf0, ef0, pm)) / (2 * eps)
        mask = np.zeros_like(fd_cf, dtype=bool)
        for s in idxs:
            n, d = divmod(s, Dm); mask[n, d] = True
        print(f"[FD head cutoff] PCC={_pcc(g_cf_np[mask], fd_cf[mask]):.4f} "
              f"max|g|={np.abs(g_cf_np[mask]).max():.3f} max|fd|={np.abs(fd_cf[mask]).max():.3f}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
