"""Choosing an operating point on a NEW photo.

Cross-photo results showed the classifier ranks candidates well but that a
score threshold learned on one photo lands in the wrong place on another
(domain shift moves the whole score distribution). Two remedies, both
usable on a photo with no annotations:

  few-shot review calibration
      The reviewer labels a small stratified sample of candidates (default
      40) as "barnacle opening" or not. A 1-D logistic (Platt) model maps
      score -> P(opening) on that sample, using inverse-probability weights
      for the stratified design, and the threshold is where P = 0.5. The
      same sample gives a design-unbiased (Horvitz-Thompson) estimate of the
      total number of openings among the candidates, with a standard error,
      independent of any threshold.
  label-free (Otsu)
      Otsu's threshold on the logit-transformed scores of the photo itself.
      Needs nothing from the reviewer, but assumes a bimodal score
      distribution.

In the evaluation the "reviewer" is simulated by the annotations (candidate
matched one-to-one to an annotation = opening), so it inherits their
incompleteness; on img2 the simulated reviewer is stricter than a real one.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

# Stratum edges as score quantiles: openings are rare among candidates, so
# the upper tail is sampled densely.
STRATA_Q = (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0)


def _feature(score, is_prob):
    s = np.asarray(score, float)
    if is_prob:
        s = np.clip(s, 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s))
    return s


def stratified_sample(score, k, rng, strata_q=STRATA_Q, legacy=False):
    """Indices of a stratified sample of EXACTLY k candidates, each item's
    weight (stratum size / number sampled from it), and the stratum of every
    candidate.

    Allocation: k // n_strata per stratum, the remainder going one each to
    the highest-score strata (where openings concentrate). A stratum smaller
    than its allocation is taken whole and the shortfall moves upward.
    `legacy=True` reproduces the old behaviour (k // n_strata per stratum,
    remainder dropped: 72 labels when 80 were requested).
    """
    edges = np.quantile(score, strata_q)
    edges[-1] += 1e-9
    strat = np.clip(np.digitize(score, edges[1:-1]), 0, len(edges) - 2)
    n_str = len(edges) - 1
    sizes = np.bincount(strat, minlength=n_str)
    k = int(min(k, len(score)))
    if legacy:
        alloc = np.minimum(np.full(n_str, max(1, k // n_str)), sizes)
    else:
        alloc = np.full(n_str, k // n_str)
        alloc[n_str - (k % n_str):] += 1 if k % n_str else 0
        alloc = np.minimum(alloc, sizes)
        short = k - alloc.sum()
        for s in range(n_str - 1, -1, -1):        # push any shortfall to strata with room
            if short <= 0:
                break
            add = min(short, sizes[s] - alloc[s])
            alloc[s] += add
            short -= add
    idx, w = [], []
    for s in range(n_str):
        if alloc[s] == 0:
            continue
        pool = np.flatnonzero(strat == s)
        take = rng.choice(pool, int(alloc[s]), replace=False)
        idx.extend(take)
        w.extend([len(pool) / len(take)] * len(take))
    return np.array(idx, int), np.array(w, float), strat


def isotonic_stratum_rates(labels, strat_of_sample, n_strata):
    """Per-stratum positive rate, forced non-decreasing with score by
    pool-adjacent-violators (weights = labels per stratum). Openings become
    more common as the classifier score rises, so an isolated positive in a
    low-score stratum is pooled with its neighbours instead of being
    multiplied by that stratum's large size."""
    n = np.bincount(strat_of_sample, minlength=n_strata).astype(float)
    pos = np.bincount(strat_of_sample, weights=labels, minlength=n_strata).astype(float)
    blocks = [[pos[s], n[s], [s]] for s in range(n_strata) if n[s] > 0]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            a, b = blocks[i], blocks.pop(i + 1)
            blocks[i] = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
            i = max(i - 1, 0)
        else:
            i += 1
    rate = np.full(n_strata, np.nan)
    for p_, n_, ss in blocks:
        rate[ss] = p_ / n_
    return rate


def isotonic_count(labels, strat_of_sample, stratum_sizes):
    """Estimated number of openings: sum over strata of size x monotone rate."""
    r = isotonic_stratum_rates(labels, strat_of_sample, len(stratum_sizes))
    return float(np.nansum(np.asarray(stratum_sizes) * r))


def label_influence(labels, strat_of_sample, stratum_sizes, estimator="isotonic"):
    """Change in the count estimate if each single label were flipped.
    Large values identify labels a reviewer should confirm."""
    labels = np.asarray(labels, int)
    f = (lambda l: isotonic_count(l, strat_of_sample, stratum_sizes)) if estimator == "isotonic" else None
    base = f(labels)
    out = np.zeros(len(labels))
    for i in range(len(labels)):
        l2 = labels.copy()
        l2[i] = 1 - l2[i]
        out[i] = f(l2) - base
    return base, out


def fit_platt(score, labels, weights, is_prob):
    x = _feature(score, is_prob)[:, None]
    if len(np.unique(labels)) < 2:
        return None
    m = LogisticRegression(C=1e4, max_iter=2000).fit(x, labels, sample_weight=weights)
    a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
    return a, b


def threshold_from_platt(ab, is_prob):
    """Score at which the calibrated probability crosses 0.5."""
    if ab is None or ab[0] <= 0:
        return None
    t = -ab[1] / ab[0]
    return float(1 / (1 + np.exp(-t))) if is_prob else float(t)


def calibrated_prob(score, ab, is_prob):
    x = _feature(score, is_prob)
    return 1 / (1 + np.exp(-np.clip(ab[0] * x + ab[1], -500, 500)))


def horvitz_thompson_total(labels, weights, strat_of_sample):
    """Estimated number of positives among all candidates, and its SE."""
    total, var = 0.0, 0.0
    for s in np.unique(strat_of_sample):
        sel = strat_of_sample == s
        Ns = weights[sel][0] * sel.sum()           # stratum size
        p = labels[sel].mean()
        n = sel.sum()
        total += Ns * p
        if n > 1:
            var += Ns ** 2 * (1 - n / Ns) * p * (1 - p) / (n - 1)
    return float(total), float(np.sqrt(var))


def otsu_threshold(score, is_prob, bins=256):
    x = _feature(score, is_prob)
    h, e = np.histogram(x, bins=bins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    t = c[np.argmax(between)]
    return float(1 / (1 + np.exp(-t))) if is_prob else float(t)
