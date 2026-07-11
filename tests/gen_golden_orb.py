"""Generate REAL-weight golden tensors for the TT-Atom Orb-v3 port (refenv only).

Run with the *reference* environment (fairchem's numpy>=2 env also has orb-models installed),
NOT the ttnn env:

    ~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py \
        --ckpt conservative-inf-omat --system bulk --out ~/.ttatom_run/goldens_real/si_omat_orb.npz

Uses the SAME Si system as tests/gen_golden_real.py's ``--system bulk`` (bulk("Si","diamond",
a=5.43)*(2,1,1), rattled stdev=0.1 seed=1) so the golden is a genuine same-system, same-task
(omat) comparison point against the already-ported UMA/eSEN backbone (si_omat.npz).

Captures the real pretrained orb-v3-conservative-inf-omat weights (downloaded from Orbital
Materials' public S3 bucket, no gating) plus per-module activations (encoder, each GNN layer,
decoder) for bottom-up PCC verification of the ttnn port.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from ase.build import bulk

from orb_models.forcefield import pretrained
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs


def npy(t):
    return t.detach().to(torch.float32).cpu().numpy() if t.dtype.is_floating_point \
        else t.detach().cpu().numpy()


def build_si():
    atoms = bulk("Si", "diamond", a=5.43) * (2, 1, 1)
    atoms.rattle(stdev=0.1, seed=1)
    return atoms


def build_short_contact():
    """A deliberately short Si-Si contact (1.4 A, well inside the ~2.2 A covalent-radii-sum ZBL
    envelope cutoff) to exercise the ZBL pair-repulsion term non-negligibly -- the existing
    ``build_si`` bulk golden's nearest-neighbor distance (2.20-2.35 A) sits just outside it (see
    docs/orb-port.md). Non-periodic (a large vacuum box) so the short contact is unambiguous."""
    from ase import Atoms

    return Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], cell=[20.0, 20.0, 20.0],
                pbc=False)


def build_molecule():
    """Baseline closed-shell molecule (water, charge=0, spin=1/singlet) -- aperiodic, same
    charge/spin convention as the UMA omol golden (tests/gen_golden_real.py --system molecule)."""
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    return atoms


def build_molecule_charged():
    """NH4+ (ammonium cation, charge=+1, spin=1/singlet) -- exercises OrbMol's charge
    conditioning with a nonzero total charge. Approximate tetrahedral geometry (not relaxed;
    Orb, like UMA, takes an arbitrary input geometry, not just equilibrium structures)."""
    from ase import Atoms

    r = 1.02
    d = r / math.sqrt(3)
    positions = [[0.0, 0.0, 0.0], [d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]]
    atoms = Atoms("NH4", positions=positions)
    atoms.info.update(charge=1, spin=1)
    return atoms


def build_molecule_openshell():
    """Methyl radical (CH3, charge=0, spin=2/doublet -- one unpaired electron) -- exercises
    OrbMol's spin conditioning with a nonzero-multiplicity open-shell system. Planar D3h
    geometry (approximate, not relaxed)."""
    from ase import Atoms

    r = 1.079
    positions = [[0.0, 0.0, 0.0]]
    for k in range(3):
        theta = 2 * math.pi * k / 3
        positions.append([r * math.cos(theta), r * math.sin(theta), 0.0])
    atoms = Atoms("CH3", positions=positions)
    atoms.info.update(charge=0, spin=2)
    return atoms


def build_si_supercell():
    """A larger periodic cell (Si diamond, (3,2,2) => 32 atoms) at production scale, big enough
    that periodic self-images (an atom connecting to its own image in a neighboring cell) occur
    within Orb's 6.0 A cutoff -- unlike the tiny 4-atom golden used for the rest of the port,
    where the ported ``tt_atom/geometry.py`` periodic graph construction (``radius_graph``) has
    not yet been exercised against Orb's own neighbor-list sign convention (see docs/orb-port.md
    Open item)."""
    atoms = bulk("Si", "diamond", a=5.43) * (3, 2, 2)
    atoms.rattle(stdev=0.05, seed=1)
    return atoms


SYSTEMS = {
    "bulk": build_si, "short_contact": build_short_contact, "supercell": build_si_supercell,
    "molecule": build_molecule, "molecule_charged": build_molecule_charged,
    "molecule_openshell": build_molecule_openshell,
}


CKPTS = {
    "conservative-inf-omat": pretrained.orb_v3_conservative_inf_omat,
    "direct-20-omat": pretrained.orb_v3_direct_20_omat,
    "conservative-omol": pretrained.orb_v3_conservative_omol,
    "direct-omol": pretrained.orb_v3_direct_omol,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="conservative-inf-omat", choices=list(CKPTS))
    ap.add_argument("--system", default="bulk", choices=list(SYSTEMS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cpu"
    orbff = CKPTS[args.ckpt](device=device, precision="float32-highest")
    orbff.eval()
    atoms = SYSTEMS[args.system]()
    graph = ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=torch.device(device))

    gns = orbff.model  # MoleculeGNS
    acts: dict[str, np.ndarray] = {}

    def save_out(name):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                for i, t in enumerate(out):
                    if torch.is_tensor(t):
                        acts[f"{name}.out{i}"] = npy(t)
            elif torch.is_tensor(out):
                acts[f"{name}.out0"] = npy(out)
        return hook

    handles = [gns._encoder.register_forward_hook(save_out("encoder"))]
    for i, blk in enumerate(gns.gnn_stacks):
        handles.append(blk.register_forward_hook(save_out(f"gnn{i}")))
    handles.append(gns._decoder.register_forward_hook(save_out("decoder")))
    if gns.conditioner is not None:
        handles.append(gns.conditioner.register_forward_hook(save_out("conditioner")))

    is_direct = args.ckpt.startswith("direct")
    result = orbff.predict(graph, split=False)
    if is_direct:
        E = float(result["energy"].item())
        F = npy(result["forces"])
        S = npy(result["stress"]) if "stress" in result else None
    else:
        E = float(result[orbff.energy_name].item())
        F = npy(result[orbff.grad_forces_name])
        S = npy(result[orbff.grad_stress_name]) if getattr(orbff, "grad_stress_name", None) else None
    print(f"orb-v3-{args.ckpt}: E={E:.6f} eV  |F|max={np.abs(F).max():.4f}")

    for h in handles:
        h.remove()

    # raw node/edge featurization (encoder inputs), for verifying the RBF+SH edge embed on device
    node_feat = gns.featurize_nodes(graph)
    edge_feat = gns.featurize_edges(graph)

    saved: dict[str, np.ndarray] = {}
    latent_dim = int(acts["encoder.out0"].shape[-1])
    cfg = dict(
        latent_dim=latent_dim,
        num_message_passing_steps=gns.num_message_passing_steps,
        edge_embed_size=gns.edge_embed_size,
        node_embed_size=gns.node_embed_size,
        lmax=gns.angular_transform._lmax,
        num_bases=gns.rbf_transform.num_bases,
        outer_product_with_cutoff=gns.outer_product_with_cutoff,
        checkpoint=args.ckpt,
        task="omol" if "omol" in args.ckpt else "omat",
        has_charge_spin_cond=gns.conditioner is not None,
    )
    saved["config"] = np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)

    saved["in@atomic_numbers"] = npy(graph.node_features["atomic_numbers"])
    saved["in@pos"] = npy(graph.node_features["positions"])
    saved["in@senders"] = npy(graph.senders)
    saved["in@receivers"] = npy(graph.receivers)
    saved["in@vectors"] = npy(graph.edge_features["vectors"])
    saved["in@cell"] = npy(graph.system_features["cell"])
    if gns.conditioner is not None:
        saved["in@charge"] = npy(graph.system_features["total_charge"])
        saved["in@spin"] = npy(graph.system_features["total_spin"])

    saved["host@node_feat"] = npy(node_feat)
    saved["host@edge_feat"] = npy(edge_feat)

    saved["out@energy"] = np.array([E], dtype=np.float64)
    saved["out@forces"] = F
    if S is not None:
        saved["out@stress"] = S

    for k, v in acts.items():
        saved[f"a@{k}"] = v

    # weights: full MoleculeGNS state dict + heads
    for k, v in gns.state_dict().items():
        saved[f"w@{k}"] = npy(v)
    for k, v in orbff.heads["energy"].state_dict().items():
        saved[f"w@energy_head.{k}"] = npy(v)
    if "forces" in orbff.heads:
        for k, v in orbff.heads["forces"].state_dict().items():
            saved[f"w@forces_head.{k}"] = npy(v)
    if "stress" in orbff.heads:
        for k, v in orbff.heads["stress"].state_dict().items():
            saved[f"w@stress_head.{k}"] = npy(v)
    if orbff.pair_repulsion:
        for k, v in orbff.pair_repulsion_fn.state_dict().items():
            saved[f"w@pair_repulsion.{k}"] = npy(v)
    if gns.conditioner is not None:
        for k, v in gns.conditioner.state_dict().items():
            saved[f"w@conditioner.{k}"] = npy(v)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **saved)
    print(f"wrote {args.out}")
    print(f"  config: {cfg}")
    print(f"  natoms={graph.senders.new_tensor(graph.node_features['atomic_numbers'].shape[0]).item()} "
          f"nedges={graph.senders.shape[0]}")
    print(f"  n weight tensors={sum(1 for k in saved if k.startswith('w@'))} n activations={len(acts)}")


if __name__ == "__main__":
    main()
