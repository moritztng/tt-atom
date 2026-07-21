"""``tt-atom`` console entry — the user-facing commands for the ttnn runtime environment.

    tt-atom run     STRUCTURE [--task] [--charge --spin] [--relax|--md] [--trace] [--out]
    tt-atom info    BUNDLE                      # config / task / weight coverage
    tt-atom verify  BUNDLE                      # device parity vs the embedded fairchem reference
    tt-atom relax   BUNDLE [--input geom.xyz | --molecule NAME] [--trace] [--fmax --steps]
    tt-atom md      BUNDLE [--input geom.xyz | --molecule NAME] [--trace] [--steps --dt --temp]
    tt-atom convert-checkpoint CKPT.pt --out BUNDLE.npz --molecule NAME [--task --charge --spin]

``run`` is the fairchem-parallel one-shot: a structure file in, energy/relax/MD out, with the
composition-specific uma-s-1 bundle auto-built (first use per composition) and cached. All commands
run here (numpy<2 + ttnn); the one-time bundle build ``run`` triggers on a cache miss shells out to
the reference (fairchem, numpy>=2) environment. ``convert-checkpoint`` is the explicit/advanced
form of that build and detects a missing fairchem, printing the exact reference-env invocation.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _resolve_charge_spin(args, bundle):
    """CLI charge/spin for a bundle-based command: an explicit --charge/--spin wins; otherwise
    default to the (charge, spin) the bundle was merged for (its embedded reference), NOT a fixed
    literal. A merged bundle bakes one charge/spin into the MoLE routing, so evaluating it with a
    mismatched runtime value silently disagrees — an omol bundle is merged at spin=1, so the old
    ``--spin`` default of 0 was wrong. Falls back to 0/0 for a reference-less bundle."""
    ref = bundle.reference
    charge = args.charge
    spin = args.spin
    if charge is None:
        charge = float(ref["charge"]) if ref is not None else 0.0
    if spin is None:
        spin = float(ref["spin"]) if ref is not None else 0.0
    return charge, spin


def _atoms(args, bundle):
    from ase.build import molecule
    if args.input:
        from ase.io import read
        atoms = read(args.input)
    else:
        atoms = molecule(args.molecule)
    charge, spin = _resolve_charge_spin(args, bundle)
    atoms.info.setdefault("charge", charge)
    atoms.info.setdefault("spin", spin)
    return atoms


def _calc(args, bundle):
    from .calculator import TTAtomCalculator
    return TTAtomCalculator(bundle, device_id=args.device_id, fast=args.fast,
                            trace=getattr(args, "trace", False))


def _run_relax(atoms, args, energy_before):
    from ase.optimize import FIRE

    FIRE(atoms, logfile="-").run(fmax=args.fmax, steps=args.steps)
    energy_after = atoms.get_potential_energy()
    fmax = float((atoms.get_forces() ** 2).sum(1).max() ** 0.5)
    print(f"relax: E {energy_before:.6f} -> {energy_after:.6f} eV; "
          f"fmax={fmax:.4f} (target {args.fmax}); converged={fmax <= args.fmax}")


def _run_md(atoms, args, energy_before=None):
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp)
    dyn = Langevin(atoms, timestep=args.dt * units.fs, temperature_K=args.temp,
                   friction=0.01 / units.fs)
    if energy_before is None:
        energy_before = atoms.get_potential_energy()

    def _log():
        ekin = atoms.get_kinetic_energy()
        print(f"  step {dyn.nsteps:4d}  E={atoms.get_potential_energy():.5f}  "
              f"T={ekin / (1.5 * units.kB * len(atoms)):.1f} K")

    dyn.attach(_log, interval=max(1, args.steps // 10))
    dyn.run(args.steps)
    print(f"md: {args.steps} steps ({args.dt} fs) at {args.temp} K; "
          f"E {energy_before:.5f} -> {atoms.get_potential_energy():.5f} eV")


def _write_output(args, atoms):
    if not args.out:
        return
    from ase.io import write

    write(args.out, atoms)
    print(f"wrote {args.out}")


def cmd_info(args):
    from .weights import WeightBundle
    b = WeightBundle.load(args.bundle)
    ok, missing, present = b.verify_coverage()
    print(f"bundle : {args.bundle}")
    print(f"task   : {b.task}")
    print(f"config : {b.config}")
    print(f"weights: {present} tensors, coverage {'OK' if ok else 'MISSING ' + str(missing[:5])}")
    print(f"scale  : rmsd={b.scale_rmsd} mean={b.scale_mean} elem_refs={'yes' if b.elem_refs is not None else 'no'}")
    ref = b.reference
    print(f"ref    : {'embedded (E=%.5f eV, %d atoms)' % (ref['energy'], len(ref['atomic_numbers'])) if ref else 'none'}")
    return 0


def cmd_verify(args):
    """Device parity vs the fairchem reference embedded in the bundle at convert time."""
    from ase import Atoms
    from .weights import WeightBundle
    from .calculator import TTAtomCalculator

    b = WeightBundle.load(args.bundle)
    ref = b.reference
    if ref is None:
        print("bundle has no embedded reference (re-export with tools/export_weights.py --uma-s-1)")
        return 2
    pbc = ref["pbc"] if ref["pbc"] is not None else False
    atoms = Atoms(numbers=ref["atomic_numbers"], positions=ref["pos"],
                  cell=ref["cell"] if ref["cell"] is not None else None, pbc=pbc)
    atoms.info.update(charge=ref["charge"], spin=ref["spin"])
    calc = TTAtomCalculator(b, device_id=args.device_id, fast=args.fast)
    atoms.calc = calc
    try:
        E = atoms.get_potential_energy()
        F = atoms.get_forces()
    finally:
        calc.close()
    Eref, Fref = ref["energy"], ref["forces"]
    rel = abs(E - Eref) / max(abs(Eref), 1e-9)
    pcc = float(np.corrcoef(F.ravel(), Fref.ravel())[0, 1])
    mae = float(np.abs(F - Fref).mean())
    ok = rel < args.etol and pcc > args.fpcc
    print(f"task={b.task}  device E={E:.5f}  ref E={Eref:.5f}  rel={rel:.2e}")
    print(f"force PCC={pcc:.5f}  MAE={mae:.3e} eV/A  |F|max dev/ref={np.abs(F).max():.4f}/{np.abs(Fref).max():.4f}")
    print("PASS" if ok else f"FAIL (need rel<{args.etol}, PCC>{args.fpcc})")
    return 0 if ok else 1


def cmd_relax(args):
    from .weights import WeightBundle
    bundle = WeightBundle.load(args.bundle) if isinstance(args.bundle, str) else args.bundle
    atoms = _atoms(args, bundle)
    calc = _calc(args, bundle)
    atoms.calc = calc
    try:
        e0 = atoms.get_potential_energy()
        _run_relax(atoms, args, e0)
        _write_output(args, atoms)
    finally:
        calc.close()
    return 0


def cmd_md(args):
    from .weights import WeightBundle
    bundle = WeightBundle.load(args.bundle) if isinstance(args.bundle, str) else args.bundle
    atoms = _atoms(args, bundle)
    calc = _calc(args, bundle)
    atoms.calc = calc
    try:
        _run_md(atoms, args)
        _write_output(args, atoms)
    finally:
        calc.close()
    return 0


def cmd_run(args):
    """One-shot: STRUCTURE -> auto-built/cached bundle -> single-point / relax / MD -> result.

    The fairchem-parallel entry point. Reads any ASE-readable structure, transparently builds and
    caches the composition-specific uma-s-1 bundle on first use (via the reference env), then runs
    on device. A cached composition needs no fairchem."""
    from ase.io import read
    from . import bundle_cache as BC
    from .calculator import TTAtomCalculator

    atoms = read(args.structure)
    atoms.info.setdefault("charge", args.charge)
    atoms.info.setdefault("spin", args.spin)
    task = args.task or BC.infer_task(atoms)     # zero-config: omat for a bulk cell, else omol
    calc = TTAtomCalculator.from_uma(model="uma-s-1", task_name=task, atoms=atoms,
                                     charge=args.charge, spin=args.spin, refenv=args.refenv,
                                     device_id=args.device_id, fast=args.fast, trace=args.trace)
    atoms.calc = calc
    try:
        e0 = atoms.get_potential_energy()
        print(f"energy: {e0:.6f} eV  ({len(atoms)} atoms, task={task}, "
              f"charge={int(args.charge)}, spin={int(args.spin)})")
        if args.relax:
            _run_relax(atoms, args, e0)
        elif args.md:
            _run_md(atoms, args, e0)
        _write_output(args, atoms)
    finally:
        calc.close()
    return 0


def cmd_convert(args):
    """Fairchem UMA checkpoint -> TT-Atom bundle. Needs the reference (fairchem) environment."""
    from .bundle_cache import exporter_path

    tools = exporter_path("export_weights.py")
    try:
        __import__("fairchem")
    except Exception:
        print("convert-checkpoint needs fairchem (numpy>=2), which cannot share this ttnn env.")
        print("Run it in the reference environment, e.g.:\n")
        print(f"  HF_HUB_OFFLINE=1 <refenv>/bin/python {tools} --uma-s-1 \\")
        print(f"      --checkpoint {args.checkpoint} --molecule {args.molecule} "
              f"--task {args.task} --charge {args.charge} --spin {args.spin} --out {args.out}")
        return 2
    import importlib.util
    spec = importlib.util.spec_from_file_location("_ttatom_export", tools)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = argparse.Namespace(uma_s_1=True, checkpoint=args.checkpoint, molecule=args.molecule,
                            task=args.task, charge=args.charge, spin=args.spin, out=args.out)
    mod.export_uma_s_1(ns)
    if args.verify:
        from .weights import WeightBundle
        ok, missing, present = WeightBundle.load(args.out).verify_coverage()
        print(f"roundtrip: reloaded bundle, coverage {'OK' if ok else 'MISSING ' + str(missing[:5])} "
              f"({present} tensors). Run `tt-atom verify {args.out}` on device for numeric parity.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tt-atom", description="TT-Atom: UMA MLIP inference on Tenstorrent")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="bf8 weights (throughput; accuracy-safe)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="one-shot: structure -> auto-bundle -> single-point/relax/md")
    p.add_argument("structure", help="ASE-readable structure (.xyz/.cif/.pdb/...)")
    # Accepted silently for compatibility with the original one-model CLI.
    p.add_argument("--uma-s-1", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--task", default=None,
                   help="dataset/task token (omol/omat/oc20/odac/omc); inferred from periodicity if unset")
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--spin", type=float, default=1.0)
    p.add_argument("--refenv", default=None, help="fairchem python for the one-time bundle build")
    p.add_argument("--trace", action="store_true", help="trace-captured device loop (~2x)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--relax", action="store_true", help="FIRE geometry relaxation")
    g.add_argument("--md", action="store_true", help="Langevin molecular dynamics")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=300.0)
    p.add_argument("--out", help="write final geometry/trajectory here")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("info", help="show bundle config/task/coverage"); p.add_argument("bundle")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("verify", help="device parity vs the bundle's embedded reference")
    p.add_argument("bundle"); p.add_argument("--etol", type=float, default=1e-3)
    p.add_argument("--fpcc", type=float, default=0.99); p.set_defaults(func=cmd_verify)

    def _sys_args(p):
        p.add_argument("bundle")
        p.add_argument("--input", help="ASE-readable geometry file (.xyz/.cif/...)")
        p.add_argument("--molecule", default="CH3CH2OH", help="ASE builtin molecule if no --input")
        p.add_argument("--charge", type=float, default=None,
                       help="net charge (default: the value the bundle was merged for)")
        p.add_argument("--spin", type=float, default=None,
                       help="spin multiplicity (default: the value the bundle was merged for)")
        p.add_argument("--trace", action="store_true", help="trace-captured device loop (~2x)")
        p.add_argument("--out", help="write final geometry here")

    p = sub.add_parser("relax", help="FIRE geometry relaxation"); _sys_args(p)
    p.add_argument("--fmax", type=float, default=0.05); p.add_argument("--steps", type=int, default=200)
    p.set_defaults(func=cmd_relax)

    p = sub.add_parser("md", help="Langevin molecular dynamics"); _sys_args(p)
    p.add_argument("--steps", type=int, default=100); p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=300.0); p.set_defaults(func=cmd_md)

    p = sub.add_parser("convert-checkpoint", help="fairchem UMA .pt -> TT-Atom bundle")
    p.add_argument("checkpoint"); p.add_argument("--out", required=True)
    p.add_argument("--molecule", default="CH3CH2OH"); p.add_argument("--task", default="omol")
    p.add_argument("--charge", type=int, default=0); p.add_argument("--spin", type=int, default=1)
    p.add_argument("--verify", action="store_true"); p.set_defaults(func=cmd_convert)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
