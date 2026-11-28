"""TensorNet end to end over one ops backend: embedding, interaction layers, readout, and the reverse pass that
returns the gradient of the summed atomic energies with respect to each edge's radial basis, cutoff and the 27
geometry scalars. The host owns the geometry and closes the chain to positions."""
from . import embed as E
from . import layer as L
from . import readout as R


def weights(model, dtype):
    return {"embed": E.weights(model.tensor_embedding, dtype), "layers": [L.weights(l, dtype) for l in model.layers],
            "readout": R.weights(model, dtype)}


def forward(o, w, z, edge, geo, send, n, eye, ol=None, edge_l=None, eye_l=None):
    """Per-atom energies [1, 1, N, 1] and the tape. edge = (rbf [S, R], cut [S, 1]). The interaction layers may run
    on a second backend `ol` (another dtype) with their own copies of the edge inputs and of eye."""
    ol, edge_l, eye_l = ol or o, edge_l or edge, eye if eye_l is None else eye_l
    x, te = E.forward(o, w["embed"], z, *edge, geo, n)
    x, tl = ol.cast(x), []
    for wl in w["layers"]:
        x, t = L.forward(ol, wl, x, *edge_l, send, n, eye_l, 1.0)
        tl.append(t)
    e, tr = R.forward(o, w["readout"], o.cast(x), eye)
    return e, (te, tl, tr)


def backward(o, w, tape, g_e, table, n, d, eye, ol=None, eye_l=None):
    """Gradient of the per-atom energies -> (gradients of rbf [S, R], one per block; of cut [S, 1], one per block;
    of geo [S, 27]). The host sums the per-block terms in its own precision."""
    ol, eye_l = ol or o, eye if eye_l is None else eye_l
    te, tl, tr = tape
    g_x = ol.cast(R.backward(o, w["readout"], tr, g_e, eye))
    g_rbf, g_cut = [], []
    for wl, t in zip(w["layers"][::-1], tl[::-1]):
        g_x, gr, gc = L.backward(ol, wl, t, g_x, table, n, d, eye_l)
        g_rbf.append(gr)
        g_cut.append(ol.usum(gc))
    gr, gc, g_geo = E.backward(o, w["embed"], te, o.cast(g_x), d)
    return g_rbf + [gr], g_cut + [gc], g_geo
