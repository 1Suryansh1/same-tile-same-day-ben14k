"""Retrieval evaluation for BEN-14K sensor agnostic retrieval.

Multi-label CBIR scores over the top-k retrieved items:
  f1_hm    : harmonic mean of the query-averaged precision and recall (CSMAE definition)
  item_f1  : per-item F1 between query and retrieved label sets, averaged over k and queries
  macro_f1 : sklearn macro F1 between query and each retrieved label vector (includes absent labels; reported only for comparison)
Options: k, query/gallery split, strict score (gallery items from the query's own acquisition removed),
self match included or excluded when query and gallery sets coincide.
"""
import argparse, numpy as np

def topk(S, k, mask=None):
    S = S.astype(np.float32, copy=True)
    if mask is not None: S[mask] = -np.inf
    part = np.argpartition(-S, k, axis=1)[:, :k]
    o = np.take_along_axis(S, part, 1).argsort(1)[:, ::-1]
    top = np.take_along_axis(part, o, 1)
    assert np.isfinite(np.take_along_axis(S, top, 1)).all(), 'mask removed too many gallery items'
    return top

def per_query(Yq, Yg, top):
    r = Yg[top]; inter = (Yq[:, None] * r).sum(-1)
    P = inter / np.maximum(r.sum(-1), 1e-12); R = inter / np.maximum(Yq.sum(-1, keepdims=True), 1e-12)
    F = np.where(inter > 0, 2 * P * R / (P + R + 1e-12), 0)
    return P.mean(1), R.mean(1), F.mean(1)

def f1_hm(P, R): p, r = P.mean(), R.mean(); return 100 * 2 * p * r / (p + r)

def macro_f1(Yq, Yg, top):
    """sklearn f1_score(query, retrieved, average='macro') per retrieved pair on the 19-dim binary vectors:
    mean of the class-1 and class-0 F1, so agreeing on ABSENT labels is rewarded (random ranker ~62)."""
    q = Yq[:, None]; r = Yg[top]; i1 = (q * r).sum(-1); i0 = ((1 - q) * (1 - r)).sum(-1)
    f1 = 2 * i1 / np.maximum(q.sum(-1) + r.sum(-1), 1e-12); d0 = (1 - q).sum(-1) + (1 - r).sum(-1)
    f0 = np.where(d0 > 0, 2 * i0 / np.maximum(d0, 1e-12), 0); return 100 * ((f1 + f0) / 2).mean()

def evaluate(Eq, Eg, Yq, Yg, k=10, acq_q=None, acq_g=None, strict=False, same_set=False, self_match='exclude', n_boot=0, seed=0):
    """Eq, Eg: L2-normalised embeddings. same_set=True when query set == gallery set (row i == row i)."""
    S = Eq @ Eg.T; mask = np.zeros(S.shape, bool)
    if same_set:
        if self_match == 'exclude': np.fill_diagonal(mask, True)
        else: S = S.copy(); np.fill_diagonal(S, np.inf)   # self ranked first
    if strict: mask |= acq_q[:, None] == acq_g[None]
    top = topk(S, k, mask if mask.any() else None)
    P, R, F = per_query(Yq, Yg, top)
    out = {'f1_hm': f1_hm(P, R), 'item_f1': 100 * F.mean(), 'macro_f1': macro_f1(Yq, Yg, top)}
    if acq_q is not None: out['own_acq_share'] = 100 * (acq_q[:, None] == acq_g[top]).mean()
    if n_boot:
        idx = np.random.default_rng(seed).integers(0, len(P), (n_boot, len(P)))
        p, r = P[idx].mean(1), R[idx].mean(1); out['f1_hm_ci95'] = list(np.percentile(100 * 2 * p * r / (p + r), [2.5, 97.5]))
    return out

def l2n(x): x = x.astype(np.float32); return x / np.linalg.norm(x, axis=1, keepdims=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Score saved embeddings (npz with val_s1, val_s2, test_s1, test_s2).')
    ap.add_argument('emb'); ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--query', default='val'); ap.add_argument('--gallery', default='test')
    ap.add_argument('--self_match', default='exclude', choices=['exclude', 'include'])
    ap.add_argument('--boot', type=int, default=0)
    a = ap.parse_args()
    import csv
    rows = list(csv.DictReader(open('splits/ben14k_patches.csv'))); Y = np.load('data/labels19_multihot.npy').astype(np.float32)
    sp = np.array([r['split_v1'] for r in rows]); acq = np.array([int(r['acquisition']) for r in rows])
    E = np.load(a.emb); qi, gi = np.where(sp == a.query)[0], np.where(sp == a.gallery)[0]
    print(f'{a.query}->{a.gallery}  k={a.k}  ' + ('self ' + a.self_match if a.query == a.gallery else ''))
    for q, g in (('s1', 's1'), ('s2', 's2'), ('s1', 's2'), ('s2', 's1')):
        for strict in (False, True):
            r = evaluate(l2n(E[f'{a.query}_{q}']), l2n(E[f'{a.gallery}_{g}']), Y[qi], Y[gi], a.k, acq[qi], acq[gi], strict, a.query == a.gallery, a.self_match, a.boot)
            print(f"  {q.upper()}->{g.upper()} {'strict  ' if strict else 'standard'} f1_hm {r['f1_hm']:6.2f}  item_f1 {r['item_f1']:6.2f}  macro_f1 {r['macro_f1']:6.2f}  own-acq {r['own_acq_share']:5.1f}%"
                  + (f"  CI [{r['f1_hm_ci95'][0]:.2f}, {r['f1_hm_ci95'][1]:.2f}]" if 'f1_hm_ci95' in r else ''))
