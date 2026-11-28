"""The TensorNet readout, forward and reverse, over the ops of tt_atom.tensornet.layer: node tensors [3, 3, N, U]
-> one energy per atom [1, 1, N, 1], in the model's own units (the caller scales and sums)."""
import torch

from . import layer as L


def weights(model, dtype=torch.float64):
    """matgl TensorNet (extensive, gated readout) -> transposed weights."""
    lin = lambda l: (l.weight.detach().to(dtype).T.contiguous(), l.bias.detach().to(dtype))
    chain = lambda seq: [lin(l) for l in seq if isinstance(l, torch.nn.Linear)]
    w = {"ln": (model.out_norm.weight.detach().to(dtype), model.out_norm.bias.detach().to(dtype)),
         "value": [lin(model.linear)] + chain(model.final_layer.gated.layers), "gate": chain(model.final_layer.gated.gates)}
    return {**w, "value_t": [wt.T.contiguous() for wt, _ in w["value"]], "gate_t": [wt.T.contiguous() for wt, _ in w["gate"]]}


def forward(o, w, x, eye):
    parts = L.decompose(o, x, eye)
    h, ln = L.norm(o, o.cat([o.csum(o.mul(p, p)) for p in parts]), *w["ln"])
    h = o.linear(h, *w["value"][0])
    val, vpre = L.mlp(o, h, w["value"][1:])
    gpre_last, gpre = L.mlp(o, h, w["gate"])
    gate = o.sigmoid(gpre_last)
    return o.mul(val, gate), dict(parts=parts, ln=ln, val=val, gate=gate, vpre=vpre, gpre=gpre)


def backward(o, w, t, g_e, eye):
    """Gradient of the per-atom energies [1, 1, N, 1] -> gradient of x."""
    gate = t["gate"]
    g_gate = o.mul(o.mul(g_e, t["val"]), o.sub(gate, o.mul(gate, gate)))
    g_h = o.add(L.mlp_t(o, o.mul(g_e, gate), w["value_t"][1:], t["vpre"]), L.mlp_t(o, g_gate, w["gate_t"], t["gpre"]))
    g_s = o.split(L.norm_t(o, o.linear(g_h, w["value_t"][0]), w["ln"][0], t["ln"]), 3)
    return L.decompose_t(o, *[o.mul(o.scale(p, 2.0), g) for p, g in zip(t["parts"], g_s)], eye)
