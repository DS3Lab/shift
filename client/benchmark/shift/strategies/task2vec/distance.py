import numpy as np
import scipy.spatial.distance as distance

_DISTANCES = {}

def _register_distance(distance_fn):
    _DISTANCES[distance_fn.__name__] = distance_fn
    return distance_fn


def is_excluded(k):
    exclude = ["fc", "linear"]
    return any([e in k for e in exclude])


def binary_entropy(p):
    from scipy.special import xlogy

    return -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p))


def get_layerwise_variance(e, normalized=False):
    var = [np.exp(l["filter_logvar"]) for l in e["layers"]]
    if normalized:
        var = [v / np.linalg.norm(v) for v in var]
    return var


def get_variance(e, normalized=False):
    var = 1.0 / np.array(e.hessian)
    if normalized:
        lambda2 = 1.0 / np.array(e.scale)
        var = var / lambda2
    return var


def get_variances(*embeddings, normalized=False):
    return [get_variance(e, normalized=normalized) for e in embeddings]


def get_hessian(e, normalized=False):
    hess = np.array(e.hessian)
    if normalized:
        scale = np.array(e.scale)
        hess = hess / scale
    return hess


def get_hessians(*embeddings, normalized=False):
    return [get_hessian(e, normalized=normalized) for e in embeddings]

def get_scaled_hessian(e0, e1):
    h0, h1 = get_hessians(e0, e1, normalized=False)
    return h0 / (h0 + h1 + 1e-8), h1 / (h0 + h1 + 1e-8)

def get_full_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = 0.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = 0.5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0, kl1

def layerwise_kl(e0, e1):
    layers0, layers1 = get_layerwise_variance(e0), get_layerwise_variance(e1)
    kl0 = []
    for var0, var1 in zip(layers0, layers1):
        kl0.append(np.sum(0.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))))
    return kl0

def layerwise_cosine(e0, e1):
    layers0, layers1 = get_layerwise_variance(
        e0, normalized=True
    ), get_layerwise_variance(e1, normalized=True)
    res = []
    for var0, var1 in zip(layers0, layers1):
        res.append(distance.cosine(var0, var1))
    return res


@_register_distance
def kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = 0.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = 0.5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return np.maximum(kl0, kl1).sum()


@_register_distance
def asymmetric_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = 0.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = 0.5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0.sum()


@_register_distance
def jsd(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    var = 0.5 * (var0 + var1)
    kl0 = 0.5 * (var0 / var - 1 + np.log(var) - np.log(var0))
    kl1 = 0.5 * (var1 / var - 1 + np.log(var) - np.log(var1))
    return (0.5 * (kl0 + kl1)).mean()


@_register_distance
def cosine(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return distance.cosine(h1, h2)


@_register_distance
def normalized_cosine(e0, e1):
    h1, h2 = get_variances(e0, e1, normalized=True)
    return distance.cosine(h1, h2)


@_register_distance
def correlation(e0, e1):
    v1, v2 = get_variances(e0, e1, normalized=False)
    return distance.correlation(v1, v2)


@_register_distance
def entropy(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return np.log(2) - binary_entropy(h1).mean()
