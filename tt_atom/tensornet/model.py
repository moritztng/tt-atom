"""TensorNet end to end over one ops backend: embedding, interaction layers, readout, and the reverse pass that
returns the gradient of the summed atomic energies with respect to each edge's radial basis, cutoff and the 27
geometry scalars. The host owns the geometry and closes the chain to positions."""
from . import embed as E
from . import layer as L
from . import readout as R


def weights(model, dtype):
    return {"embed": E.weights(model.tensor_embedding, dtype), "layers": [L.weights(l, dtype) for l in model.layers],
            "readout": R.weights(model, dtype)}


def forward(o, w, z, rbf, cut, geo, send, n, eye):
    """Per-atom energies [1, 1, N, 1] and the tape."""
    x, te = E.forward(o, w["embed"], z, rbf, cut, geo, n)
    tl = []
    for wl in w["layers"]:
        x, t = L.forward(o, wl, x, rbf, cut, send, n, eye, 1.0)
        tl.append(t)
    e, tr = R.forward(o, w["readout"], x, eye)
    return e, (te, tl, tr)


def backward(o, w, tape, g_e, table, n, d, eye):
    """Gradient of the per-atom energies -> gradients of (rbf [S, R], cut [S, 1], geo [S, 27])."""
    te, tl, tr = tape
    g_x = R.backward(o, w["readout"], tr, g_e, eye)
    g_rbf = g_cut = None
    for wl, t in zip(w["layers"][::-1], tl[::-1]):
        g_x, gr, gc = L.backward(o, wl, t, g_x, table, n, d, eye)
        gc = o.usum(gc)
        g_rbf, g_cut = (gr, gc) if g_rbf is None else (o.add(g_rbf, gr), o.add(g_cut, gc))
    gr, gc, g_geo = E.backward(o, w["embed"], te, g_x, d)
    return o.add(g_rbf, gr), o.add(g_cut, gc), g_geo
