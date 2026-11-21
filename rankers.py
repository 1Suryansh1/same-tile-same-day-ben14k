"""Trivial reference rankers (no learned features). Each returns per-query expected (P, R) or a score matrix.
random      : exact expectation of a uniformly random ranking over the allowed gallery
acq_only    : gallery items from the query's own acquisition first, random order inside each tier
              (expectation = random over own acquisition; under the strict score it equals random over the rest)
location    : rank by Chebyshev distance on the patch grid inside the same acquisition (no pixels), ties broken by random jitter.
              Scores the same in every direction: the S1 and S2 patches of a pair share location by construction.
band_stats  : Euclidean distance on per-band mean, std and 10/50/90th percentiles (standardised). Within-sensor only.
"""
import csv, numpy as np
from eval import per_query, f1_hm, topk

def expected_PR(Yq, Yg, allowed):
    inter = Yq @ Yg.T; P = inter / Yg.sum(1)[None]; R = inter / Yq.sum(1)[:, None]; a = allowed.astype(np.float32)
    return (P * a).sum(1) / a.sum(1), (R * a).sum(1) / a.sum(1)

def location_scores(row_q, col_q, acq_q, row_g, col_g, acq_g, seed=0):
    d = np.maximum(np.abs(row_q[:, None] - row_g[None]), np.abs(col_q[:, None] - col_g[None])).astype(np.float32)
    d[acq_q[:, None] != acq_g[None]] = 1e4
    return -d + 1e-3 * np.random.default_rng(seed).random(d.shape)

if __name__ == '__main__':
    rows = list(csv.DictReader(open('splits/ben14k_patches.csv'))); Y = np.load('data/labels19_multihot.npy').astype(np.float32)
    sp = np.array([r['split_v1'] for r in rows]); acq = np.array([int(r['acquisition']) for r in rows])
    gr = np.array([int(r['grid_row']) for r in rows]); gc = np.array([int(r['grid_col']) for r in rows])
    qi, gi = np.where(sp == 'val')[0], np.where(sp == 'test')[0]; same = acq[qi][:, None] == acq[gi][None]
    for name, std_allowed, strict_allowed in (('random', np.ones_like(same), ~same), ('acquisition only', same, ~same)):
        print(f'{name:18s} standard {f1_hm(*expected_PR(Y[qi], Y[gi], std_allowed)):6.2f}  strict {f1_hm(*expected_PR(Y[qi], Y[gi], strict_allowed)):6.2f}')
    L = location_scores(gr[qi], gc[qi], acq[qi], gr[gi], gc[gi], acq[gi])
    for strict in (False, True):
        P, R, _ = per_query(Y[qi], Y[gi], topk(L, 10, same if strict else None)); print(f"location           {'strict' if strict else 'standard'} {f1_hm(P, R):6.2f}")
    B = np.load('data/bandstats_valtest.npz'); nq = len(qi)
    for m in ('s2', 's1'):
        F = B[m]; F = (F - F.mean(0)) / (F.std(0) + 1e-6); Fq, Fg = F[:nq], F[nq:]
        S = -((Fq ** 2).sum(1)[:, None] - 2 * Fq @ Fg.T + (Fg ** 2).sum(1)[None])
        for strict in (False, True):
            P, R, _ = per_query(Y[qi], Y[gi], topk(S, 10, same if strict else None)); print(f"band stats {m.upper()}->{m.upper()}  {'strict' if strict else 'standard'} {f1_hm(P, R):6.2f}")
