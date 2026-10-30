"""
Additional analyses.

  A. Bootstrap confidence intervals for ARI from the cached manifolds.
  B. K-Means at K = 2..10 with the dominant sub-type of every cluster.
  C. Sensitivity to the kernel width, the number of Fourier directions and the
     UMAP neighbourhood size.

Output: results/sensitivity/
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import framework as fw  # noqa: E402
CACHE = os.path.join(fw.DATA, "_single_setup_cache")
OUT = os.path.join(fw.RES, "sensitivity")
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 43, 44]
K = 4
B = 1000
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(LOG))


def purity(y, p):
    cm = pd.crosstab(p, y).values
    return cm.max(axis=1).sum() / cm.sum()


def load():
    from rhetorical_features import clean_text_aggressive
    df = pd.read_csv(fw.CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    df = df.reset_index(drop=True)
    return df, clean_text_aggressive


def part_a(y):
    say(f"=== A. Bootstrap 95% confidence intervals for ARI (K-Means, cached manifolds, B = {B}) ===")
    conds = ["fused", "semantic", "rhetorical", "rhetorical_raw", "fused_norbf", "artefact", "length"]
    preds = {}
    for c in conds:
        preds[c] = []
        for s in SEEDS:
            U = np.load(os.path.join(CACHE, f"U_{c}_s{s}.npy"))
            preds[c].append(KMeans(K, n_init=10, random_state=s).fit_predict(U))
    rng = np.random.RandomState(0)
    n = len(y)
    boots = {c: np.zeros(B) for c in conds}
    for b in range(B):
        idx = rng.randint(0, n, n)
        for c in conds:
            boots[c][b] = np.mean([adjusted_rand_score(y[idx], p[idx]) for p in preds[c]])
    rows = []
    for c in conds:
        point = np.mean([adjusted_rand_score(y, p) for p in preds[c]])
        lo, hi = np.percentile(boots[c], [2.5, 97.5])
        rows.append({"condition": c, "ARI": round(point, 4), "CI_low": round(lo, 4), "CI_high": round(hi, 4)})
        say(f"  {c:<16} ARI {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "A_bootstrap_ci.csv"), index=False)
    drows = []
    for other in ["semantic", "fused_norbf", "length", "artefact", "rhetorical"]:
        d = boots["fused"] - boots[other]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = (d <= 0).mean()
        drows.append({"comparison": f"fused - {other}", "diff_mean": round(d.mean(), 4), "CI_low": round(lo, 4),
                      "CI_high": round(hi, 4), "share_of_boots_with_diff<=0": p})
        say(f"  fused - {other:<12} diff {d.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]  bootstrap p = {p:.4f}")
    pd.DataFrame(drows).to_csv(os.path.join(OUT, "A_bootstrap_differences.csv"), index=False)


def part_b(y):
    say("\n=== B. Cluster structure for K = 2..10 (fused 1.5:1, K-Means, seed 42) ===")
    U = np.load(os.path.join(CACHE, "U_fused_s42.npy"))
    rows, summ = [], []
    for k in range(2, 11):
        p = KMeans(k, n_init=10, random_state=42).fit_predict(U)
        ct = pd.crosstab(p, y)
        shares = ct.max(axis=1) / ct.sum(axis=1)
        dom = ct.idxmax(axis=1)
        for c in ct.index:
            rows.append({"K": k, "cluster": int(c), "size": int(ct.loc[c].sum()), "dominant": dom[c],
                         "dominant_share": round(float(shares[c]), 4)})
        mixed = int((shares < 0.70).sum())
        per_type = dom.value_counts().to_dict()
        summ.append({"K": k, "purity": round(purity(y, p), 4), "NMI": round(normalized_mutual_info_score(y, p), 4),
                     "ARI": round(adjusted_rand_score(y, p), 4), "clusters_below_70pct": mixed,
                     "clusters_per_subtype": per_type})
        say(f"  K={k:2d}  purity {purity(y, p):.4f}  NMI {normalized_mutual_info_score(y, p):.4f}  "
            f"mixed clusters (<70% one sub-type): {mixed}  clusters per sub-type: {per_type}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "B_clusters_per_K.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "B_summary_per_K.csv"), index=False)


def part_c(df, clean, y):
    say("\n=== C. Sensitivity to D, gamma and UMAP neighbours (fused 1.5:1, K-Means, seeds 42-44) ===")
    import umap.umap_ as umap
    F, cols = fw.load_features(df, False, [], refresh=False)
    E = fw.sbert(df["text"].map(clean), f"cased_{len(df)}")
    S = normalize(np.asarray(E, dtype=np.float32))
    Rraw = MinMaxScaler().fit_transform(F[cols].to_numpy(float))
    configs = [("reported (D=200, gamma=1, nn=15)", 200, 1.0, 15),
               ("D = 100", 100, 1.0, 15), ("D = 400", 400, 1.0, 15),
               ("gamma = 0.5", 200, 0.5, 15), ("gamma = 2", 200, 2.0, 15),
               ("UMAP neighbours = 10", 200, 1.0, 10), ("UMAP neighbours = 30", 200, 1.0, 30)]
    rows = []
    for name, D, g, nn in configs:
        R = fw.rff(Rraw, D=D, g=g)
        X = normalize(np.hstack([S * 1.5, R * 1.0]))
        res = []
        for s in SEEDS:
            t0 = time.time()
            if (D, g, nn) == (200, 1.0, 15):
                U = np.load(os.path.join(CACHE, f"U_fused_s{s}.npy"))
            else:
                Z = TruncatedSVD(fw.SVD_DIMS, random_state=s).fit_transform(X)
                U = umap.UMAP(n_neighbors=nn, n_components=fw.UMAP_DIMS, min_dist=0.0, metric="cosine",
                              random_state=s, low_memory=True, n_epochs=fw.UMAP_EPOCHS, n_jobs=1).fit_transform(Z)
            p = KMeans(K, n_init=10, random_state=s).fit_predict(U)
            res.append((adjusted_rand_score(y, p), normalized_mutual_info_score(y, p), purity(y, p)))
            say(f"  {name:<34} seed {s}  ARI {res[-1][0]:.4f}  [{time.time()-t0:.0f}s]")
        a = np.array(res)
        rows.append({"setting": name, "ARI_mean": round(a[:, 0].mean(), 4), "ARI_sd": round(a[:, 0].std(ddof=1), 4),
                     "NMI_mean": round(a[:, 1].mean(), 4), "Purity_mean": round(a[:, 2].mean(), 4)})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "C_sensitivity.csv"), index=False)
    say("\n" + pd.DataFrame(rows).to_string(index=False))


def main():
    t0 = time.time()
    df, clean = load()
    y = df["label"].to_numpy()
    say(f"corpus: {len(df):,} documents")
    part_a(y)
    part_b(y)
    part_c(df, clean, y)
    say(f"\ntotal time {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
