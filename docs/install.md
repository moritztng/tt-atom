# Installation details

## Source tt-metal

`install_dependencies.sh` installs the host packages and SFPI compiler required by
`build_metal.sh`. Run it once before the first source build. Keep `TT_METAL_HOME` exported at
runtime and retain `$TT_METAL_HOME/build_Release`, where JIT-compiled kernels are loaded.

If a kernel compile fails with `riscv-tt-elf-g++: error: unrecognized option '-ftt-...'`, update
SFPI and clear the old kernel cache before rebuilding:

```bash
sudo ./install_dependencies.sh --sfpi
rm -rf ~/.cache/tt-metal-cache/
```

Some board and firmware combinations require an explicit mesh descriptor:

```bash
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

Set it before constructing `tt_atom.batch.MultiCard` so worker processes inherit it.

## Bundles and reference dependencies

`Calculator(atoms)` exports and caches the required model weights on first use. UMA caches one
bundle per reduced composition, charge, spin, and task. Orb caches one bundle per checkpoint.

The exporters use a separate reference environment because fairchem requires `numpy>=2` while
the validated `ttnn` environment uses `numpy<2`:

```bash
python -m venv refenv
refenv/bin/pip install "fairchem-core>=2.10" "orb-models==0.5.5"
```

Pass `refenv="/path/to/refenv/bin/python"` to `Calculator`, or set `TT_ATOM_REFENV`. Cached runs
do not need the reference environment.

`orb-models==0.5.5` is required by the current exporter. Version 0.6 changed the pretrained
checkpoint API and moved `system_config`.

For a manual export from a source checkout:

```bash
refenv/bin/python tools/export_weights.py \
  --uma-s-1 --xyz structure.xyz --task omol --out model.npz
refenv/bin/python tools/export_orb_weights.py \
  --ckpt conservative-inf-omat --out weights.npz
```

Load the resulting file with `TTAtomCalculator("model.npz")` or
`OrbCalculator("weights.npz")`.
