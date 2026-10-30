"""
Ablation, stream-weight sweep, confusion matrix and choice of K, all from one
setup (seeds 42, 43, 44; each seed changes the SVD/UMAP fit and the clustering).

Manifolds are cached in data/_single_setup_cache/, so the script resumes if stopped.
Output: results/single_setup/
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import framework as fw

SEEDS = [42, 43, 44]
K = 4
WS, WR = 1.5, 1.0
GRID = [(1, 0), (5, 1), (4, 1), (3, 1), (2, 1), (1.5, 1), (1, 1), (1, 1.5), (1, 2), (1, 3), (0, 1)]
OUT = os.path.join(fw.RES, "single_setup")
CACHE = os.path.join(fw.DATA, "_single_setup_cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def manifold(X, name, seed):
    """SVD(100) -> UMAP(40), with the seed applied to both steps. Cached."""
    path = os.path.join(CACHE, f"U_{name}_s{seed}.npy")
    if os.path.exists(path):
        return np.load(path)
    if X.shape[1] <= 2:                      # length baseline: no manifold
        U = X
    else:
        import umap.umap_ as umap
        Z = TruncatedSVD(min(fw.SVD_DIMS, X.shape[1] - 1), random_state=seed).fit_transform(X)
        U = umap.UMAP(n_neighbors=fw.UMAP_NEIGH, n_components=min(fw.UMAP_DIMS, Z.shape[1] - 1),
                      min_dist=0.0, metric="cosine", random_state=seed, low_memory=True,
                      n_epochs=fw.UMAP_EPOCHS, n_jobs=1).fit_transform(Z)
    np.save(path, U)
    return U


def main():
    from rhetorical_features import clean_text_aggressive
    t00 = time.time()
    df = pd.read_csv(fw.CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    df = df.reset_index(drop=True)
    y = df["label"].to_numpy()
    classes = [c for c in fw.CLASSES if (y == c).any()]
    say(f"corpus {os.path.basename(fw.CORPUS)}: {len(df):,} documents")
    if df["text"].str.contains(r"\[SEP\]").any():
        sys.exit("The corpus still contains [SEP]. Run 01_clean_corpus.py first.")

    F, cols = fw.load_features(df, False, [], refresh=False)
    E = fw.sbert(df["text"].map(clean_text_aggressive), f"cased_{len(df)}")
    S = normalize(np.asarray(E, dtype=np.float32))
    Rraw = MinMaxScaler().fit_transform(F[cols].to_numpy(float))
    R = fw.rff(Rraw)
    ART = normalize(MinMaxScaler().fit_transform(fw.artefact_features(df["text"])))
    LEN = MinMaxScaler().fit_transform(df["text"].str.split().str.len().to_numpy().reshape(-1, 1))

    def fused(ws, wr, rh=R):
        if wr == 0:
            return S
        if ws == 0:
            return rh
        return normalize(np.hstack([S * ws, rh * wr]))

    conds = {  # Table VIII
        "fused": ("Fused, ws:wr = 1.5:1", fused(WS, WR)),
        "semantic": ("Semantic stream only", S),
        "rhetorical": ("Rhetorical stream only (RBF expanded)", R),
        "rhetorical_raw": ("Rhetorical stream only (raw features)", Rraw),
        "fused_norbf": ("Fused without the RBF expansion", fused(WS, WR, Rraw)),
        "artefact": ("Artefact baseline", ART),
        "length": ("Document length alone", LEN),
    }
    for ws, wr in GRID:  # Table IX
        key = {(1, 0): "semantic", (0, 1): "rhetorical", (1.5, 1): "fused"}.get((ws, wr), f"w{ws:g}-{wr:g}")
        if key not in conds:
            conds[key] = (f"Fused, ws:wr = {ws:g}:{wr:g}", fused(ws, wr))

    rows = []
    for key, (name, X) in conds.items():
        for seed in SEEDS:
            t0 = time.time()
            U = manifold(X, key, seed)
            km = KMeans(K, n_init=10, random_state=seed).fit_predict(U)
            gm = fw.gmm(U, K, seed)
            rows.append(fw.score(y, km, classes, condition=key, name=name, dim=X.shape[1], seed=seed, algorithm="kmeans"))
            rows.append(fw.score(y, gm, classes, condition=key, name=name, dim=X.shape[1], seed=seed, algorithm="gmm"))
            say(f"  {name:<42} seed {seed}  ARI km {rows[-2]['ARI']:.4f}  gmm {rows[-1]['ARI']:.4f}  [{time.time()-t0:.0f}s]")
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_run_scores.csv"), index=False)

    P = pd.DataFrame(rows)
    agg = (P.groupby(["condition", "name", "dim", "algorithm"])[["ARI", "NMI", "Purity", "Match %"]]
             .agg(["mean", "std"]).round(4))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    t8 = agg[agg.condition.isin(["fused", "semantic", "rhetorical", "rhetorical_raw", "fused_norbf", "artefact", "length"])]
    t8.to_csv(os.path.join(OUT, "t08_ablation.csv"), index=False)

    order = []
    for ws, wr in GRID:
        key = {(1, 0): "semantic", (0, 1): "rhetorical", (1.5, 1): "fused"}.get((ws, wr), f"w{ws:g}-{wr:g}")
        sub = agg[agg.condition == key].copy()
        sub.insert(0, "w_s:w_r", f"{ws:g}:{wr:g}")
        order.append(sub)
    t9 = pd.concat(order)
    t9.to_csv(os.path.join(OUT, "t09_weight_sweep.csv"), index=False)

    # the same numbers in the format read by 12_figures.py
    def wide(sub):
        out = {}
        for alg in ("gmm", "kmeans"):
            r = sub[sub.algorithm == alg].iloc[0]
            out.update({f"ARI_{alg}": r.ARI_mean, f"ARI_{alg}_sd": r.ARI_std, f"NMI_{alg}": r.NMI_mean,
                        f"Purity_{alg}": r.Purity_mean, f"Match_{alg}": r["Match %_mean"]})
        return out
    pd.DataFrame([{"condition": c, "name": conds[c][0], "dim": conds[c][1].shape[1], "seconds": 0,
                   **wide(agg[agg.condition == c])}
                  for c in ["fused", "semantic", "rhetorical", "rhetorical_raw", "fused_norbf", "artefact", "length"]]
                 ).to_csv(os.path.join(fw.RES, f"ablation_k{K}.csv"), index=False)
    pd.DataFrame([{"w_s": ws, "w_r": wr, "seconds": 0,
                   **wide(agg[agg.condition == {(1, 0): "semantic", (0, 1): "rhetorical", (1.5, 1): "fused"}
                              .get((ws, wr), f"w{ws:g}-{wr:g}")])} for ws, wr in GRID]
                 ).to_csv(os.path.join(fw.RES, f"weight_sweep_published_k{K}.csv"), index=False)

    # Table XI: the confusion matrix of seed 42, K-Means, fused
    U = manifold(conds["fused"][1], "fused", 42)
    km = KMeans(K, n_init=10, random_state=42).fit_predict(U)
    al = fw.align(y, km, classes)
    cm = pd.crosstab(pd.Series(al, name="cluster"), pd.Series(y, name="true"))
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)
    cm.to_csv(os.path.join(OUT, "t11_confusion.csv"))
    match42 = 100 * np.trace(cm.to_numpy()) / cm.to_numpy().sum()

    # Table XII: choice of K on the same manifold (fused, seed 42)
    ks = []
    for k in range(2, 11):
        g = GaussianMixture(k, covariance_type="diag", n_init=5, reg_covar=1e-4, random_state=42).fit(U)
        pg = g.predict(U)
        pk = KMeans(k, n_init=10, random_state=42).fit_predict(U)
        ks.append({"K": k, "BIC_gmm": round(g.bic(U), 1),
                   "silhouette_kmeans": round(float(silhouette_score(U, pk, sample_size=min(5000, len(U)), random_state=42)), 4),
                   "silhouette_gmm": round(float(silhouette_score(U, pg, sample_size=min(5000, len(U)), random_state=42)), 4),
                   "ARI_kmeans": round(adjusted_rand_score(y, pk), 4), "ARI_gmm": round(adjusted_rand_score(y, pg), 4)})
        say(f"  K={k}  BIC {ks[-1]['BIC_gmm']}  silhouette(km) {ks[-1]['silhouette_kmeans']}  ARI km {ks[-1]['ARI_kmeans']}")
    pd.DataFrame(ks).to_csv(os.path.join(OUT, "t12_k_selection.csv"), index=False)

    f = agg[(agg.condition == "fused") & (agg.algorithm == "kmeans")].iloc[0]
    say("\n================ SUMMARY (cleaned corpus, one setup, seeds 42-44 incl. UMAP) ================")
    for c in ["fused", "semantic", "rhetorical", "fused_norbf", "artefact", "length"]:
        r = agg[(agg.condition == c) & (agg.algorithm == "kmeans")].iloc[0]
        g = agg[(agg.condition == c) & (agg.algorithm == "gmm")].iloc[0]
        say(f"  {r['name']:<42} ARI km {r.ARI_mean:.4f} ± {r.ARI_std:.4f}   gmm {g.ARI_mean:.4f} ± {g.ARI_std:.4f}   match km {r['Match %_mean']:.2f}")
    say(f"\n  Table XI (seed 42) match: {match42:.2f}%   (mean over seeds: {f['Match %_mean']:.2f}%)")
    best = t9[t9.algorithm == "kmeans"].sort_values("ARI_mean", ascending=False).iloc[0]
    say(f"  best weighting in the sweep (K-Means): {best['w_s:w_r']}  ARI {best.ARI_mean:.4f}")
    say(f"  total time {(time.time()-t00)/60:.0f} min")
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    say(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
