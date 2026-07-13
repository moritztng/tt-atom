"""``Calculator`` — the single public entry point across both model families.

One call, model picked by name, no need to know whether it is UMA or Orb (mirrors fairchem's
``FAIRChemCalculator`` and Hugging Face's ``AutoModel.from_pretrained``)::

    from tt_atom import Calculator
    atoms.calc = Calculator(atoms)                              # uma-s-1 (the default)
    atoms.calc = Calculator(atoms, "orb-v3-conservative-omol")  # an Orb checkpoint

The name selects the family. ``uma-*`` routes to the equivariant eSCN-MD engine
(:meth:`tt_atom.calculator.TTAtomCalculator.from_uma`, task inferred from periodicity);
``orb-*`` to the Orb-v3/OrbMol backbone
(:meth:`tt_atom.orb_calculator.OrbCalculator.from_checkpoint`, where the name *is* the checkpoint).
Everything downstream is plain ASE either way.

The two families genuinely differ, so a family-inapplicable argument raises rather than being
silently dropped: ``task`` exists only for UMA, and ``charge``/``spin`` condition every UMA task
but only the OrbMol checkpoints (the Orb-v3 ``omat`` checkpoints were never trained with
conditioning and ignore them, as documented). Reach for the classes / classmethods directly
(``TTAtomCalculator``, ``OrbCalculator``, ``.from_uma``, ``.from_checkpoint``) only when you want
to pin a UMA task, hand in a prebuilt bundle, or manage the weight file yourself.
"""
from __future__ import annotations


def _family(model):
    """Map a model name to its family (``"uma"`` / ``"orb"``); raise on anything else."""
    m = str(model).lower()
    if m.startswith("uma"):
        return "uma"
    if m.startswith("orb"):
        return "orb"
    from . import orb_weight_cache as OWC

    raise ValueError(
        f"unknown model {model!r}: expected a UMA model (e.g. 'uma-s-1') or an Orb checkpoint "
        f"(one of {', '.join(OWC.CHECKPOINTS)}).")


def Calculator(atoms=None, model="uma-s-1", *, task=None, charge=0, spin=1, refenv=None,
               checkpoint=None, cache_dir=None, device=None, device_id=0, fast=False,
               trace=False, **kwargs):
    """Return a ready ASE calculator for ``model``, dispatching by name across both families.

    ``model`` is the single selector — ``"uma-s-1"`` (the default) for UMA, or one of the four
    Orb checkpoint names (``"orb-v3-{conservative,direct}-{omat,omol}"``, see
    ``tt_atom.orb_weight_cache.CHECKPOINTS``). ``atoms`` is required for UMA (its composition fixes
    the merged bundle) and optional for Orb (only used to stamp ``charge``/``spin`` defaults).
    First use of a given model builds and caches its weights via the reference env; later calls are
    a plain load. Every other keyword is passed through to the family entry point it applies to."""
    if _family(model) == "uma":
        from . import bundle_cache as BC
        from .calculator import TTAtomCalculator

        if atoms is None:
            raise ValueError(
                "a UMA model needs `atoms` to determine the composition — MoLE bakes one bundle "
                "per reduced composition/charge/spin/task, so there is no bundle to pick or build "
                "without the structure. Pass the Atoms you want to run.")
        if task is None:
            task = BC.infer_task(atoms)
        return TTAtomCalculator.from_uma(model=model, task_name=task, atoms=atoms, charge=charge,
                                         spin=spin, refenv=refenv, checkpoint=checkpoint,
                                         cache_dir=cache_dir, device=device, device_id=device_id,
                                         fast=fast, trace=trace, **kwargs)

    # Orb family: the model name IS the checkpoint; no task, no per-composition bundle.
    from .orb_calculator import OrbCalculator

    if task is not None:
        raise ValueError(
            f"`task` applies only to UMA models; the Orb checkpoint {model!r} has no task selector "
            "(the checkpoint itself fixes omat vs omol). Drop task=.")
    if checkpoint is not None:
        raise ValueError(
            f"for Orb the model name is the checkpoint (model={model!r}); there is no separate "
            "raw-checkpoint override to pass. Drop checkpoint=.")
    if trace:
        raise ValueError(
            "trace= is not wired into the Orb calculator; capture a fixed-topology Orb loop with "
            "tt_atom.orb_trace.OrbTracedEngine directly (see examples/orb_md.py).")
    # charge/spin condition only the OrbMol checkpoints; the omat checkpoints ignore them (no
    # conditioning weights). Stamp defaults onto atoms exactly as UMA does, so `calculate` reads
    # back the same values — a harmless no-op for a non-conditioned checkpoint.
    if atoms is not None:
        atoms.info.setdefault("charge", charge)
        atoms.info.setdefault("spin", spin)
    return OrbCalculator.from_checkpoint(checkpoint=model, refenv=refenv, cache_dir=cache_dir,
                                         device=device, device_id=device_id, fast=fast, **kwargs)
