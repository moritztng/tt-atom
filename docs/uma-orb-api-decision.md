# UMA and Orb public API

## Decision

Keep `UMA(atoms, ...)` and `Orb(atoms, ...)` as the public constructors. Both return an ASE
calculator, so usage after construction stays identical.

## Why

The constructors have different contracts:

- UMA requires `atoms` because its weights are merged and cached for a composition, charge, spin,
  and task. Orb can create a calculator before a structure is available because its cache is keyed
  only by checkpoint.
- `task` and calculator-level tracing apply to UMA. Orb selects `omat` or `omol` through its
  checkpoint and exposes tracing through a separate engine.
- `model` selects a UMA release, while Orb's corresponding selector is the checkpoint itself.
- Charge and spin condition every UMA task, but only the OrbMol checkpoints.

`Calculator(atoms, model=...)` hid these differences behind one broad signature. It then needed
runtime checks for combinations that its signature appeared to accept. Separate names make the
selected family and its valid options visible in the call and in editor signature help.

The familiar auto-model pattern does not outweigh that cost. Hugging Face auto classes normalize
one operation, loading a model from an identifier. TT-Atom's choice changes construction
requirements and valid controls. The installed fairchem `FAIRChemCalculator` also takes an
already-created prediction unit plus a task, rather than dispatching model families from a name.

Cache implementation alone would not justify separate APIs. The user-visible differences in
required inputs and supported controls do. If future families converge on one construction
contract, a shared factory can be reconsidered without changing the ASE calculator protocol.

## Result

The unified public-API branch is not merged. The shared internal calculator lifecycle from the
earlier refactor remains, while the README and package exports continue to present `UMA()` and
`Orb()`.
