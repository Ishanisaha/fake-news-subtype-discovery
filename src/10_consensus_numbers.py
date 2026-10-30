"""
Final table values for the consensus version from the cached manifolds of
08_draw_stability.py and 09_consensus_tables.py, plus bootstrap confidence
intervals (600 resamples).

    python 10_consensus_numbers.py          # table values
    python 10_consensus_numbers.py boot     # bootstrap intervals

Output: results/consensus/consensus_numbers.json
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from scipy.optimize import linear_sum_assignment
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
C1 = os.path.join(RES, "consensus", "cache")
C2 = os.path.join(RES, "draw_stability", "cache")
OLD = os.path.join(DATA, "_single_setup_cache")
OUT = os.path.join(RES, "consensus")
K = 4
DR = range(42, 52)
df = pd.read_csv(os.path.join(DATA, "corpus.csv"))
y = df["label"].replace({"rumor": "rumour"}).to_numpy()
CL = ["propaganda", "rumour", "satire", "clickbait"]
def U(tag):
    for c in (C1, C2):
        p = os.path.join(c, f"U_{tag}_s42.npy")
        if os.path.exists(p): return np.load(p)
    raise FileNotFoundError(tag)
def pur(y, p): cm = pd.crosstab(p, y).values; return cm.max(1).sum() / cm.sum()
def match(y, p):
    cm = pd.crosstab(p, y).values; r, c = linear_sum_assignment(-cm); return cm[r, c].sum() / cm.sum()
def cons(parts, s): return KMeans(K, n_init=20, random_state=s).fit_predict(np.hstack([np.eye(K)[q] for q in parts]))
def gmm(u): return GaussianMixture(K, covariance_type="diag", n_init=5, reg_covar=1e-4, random_state=42).fit_predict(u)
res = {}
def row(name, tags):
    us = [U(t) for t in tags]
    km = [KMeans(K, n_init=10, random_state=42).fit_predict(u) for u in us]
    gm = [gmm(u) for u in us]
    P = [cons(km, s) for s in (42, 43, 44)]
    G = cons(gm, 42)
    r = {"ARI_KM": np.mean([ari(y, p) for p in P]), "ARI_KM_sd": np.std([ari(y, p) for p in P], ddof=1),
         "ARI_GMM": ari(y, G), "NMI": np.mean([nmi(y, p) for p in P]), "Pur": np.mean([pur(y, p) for p in P]),
         "Match": 100 * np.mean([match(y, p) for p in P]),
         "single_KM_mean": np.mean([ari(y, p) for p in km]), "single_KM_min": min(ari(y, p) for p in km),
         "single_KM_max": max(ari(y, p) for p in km)}
    res[name] = {k: round(float(v), 4) for k, v in r.items()}
    print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(OUT, "consensus_numbers.json"), "w"), indent=1)
    return P[0]
if len(sys.argv) > 1 and sys.argv[1] == "boot":
    us = [U(f"draw{d}") for d in DR]
    pf = cons([KMeans(K, n_init=10, random_state=42).fit_predict(u) for u in us], 42)
    res = json.load(open(os.path.join(OUT, "consensus_numbers.json")))
else:
    pf = row("fused_1.5-1", [f"draw{d}" for d in DR])
    ct = pd.crosstab(pf, y)[CL]; ct.to_csv(os.path.join(OUT, "confusion_consensus_s42.csv"))
    row("rhetorical_exp", [f"rhet_draw{d}" for d in DR])
    for w in ["5-1", "4-1", "3-1", "2-1", "1-1", "1-1.5", "1-2", "1-3"]:
        row("w" + w, [f"w{w}_draw{d}" for d in DR])
    sys.exit(0)
# bootstrap: consensus fused vs cached single-setup baselines (3 seeds)
base = {c: [KMeans(K, n_init=10, random_state=s).fit_predict(np.load(os.path.join(OLD, f"U_{c}_s{s}.npy"))) for s in (42, 43, 44)]
        for c in ["semantic", "fused_norbf", "length", "artefact"]}
rng = np.random.RandomState(0); n = len(y); B = 600
bt = {"fused": np.zeros(B), **{c: np.zeros(B) for c in base}}
for b in range(B):
    i = rng.randint(0, n, n)
    bt["fused"][b] = ari(y[i], pf[i])
    for c in base: bt[c][b] = np.mean([ari(y[i], p[i]) for p in base[c]])
out = {"fused_CI": [round(float(x), 4) for x in np.percentile(bt["fused"], [2.5, 97.5])]}
for c in base:
    d = bt["fused"] - bt[c]
    out[f"fused-{c}"] = [round(float(d.mean()), 4)] + [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])] + [float((d <= 0).mean())]
res["bootstrap"] = out; print(out)
json.dump(res, open(os.path.join(OUT, "consensus_numbers.json"), "w"), indent=1)
print("DONE")
