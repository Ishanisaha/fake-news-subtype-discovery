"""
Dependence on the random Fourier draw.

  P1. 20 random Fourier draws (seeds 42-61) at 1.5 : 1.
  P2. Consensus over draws 42-51 and over draws 52-61, and their agreement.
  P3. Choice of the draw by silhouette.
  P4. Variant without the kernel expansion (no random draw).

Output: results/draw_stability/
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import framework as fw  # noqa: E402
OUT = os.path.join(fw.RES, "draw_stability")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
K = 4
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(LOG))


def purity(y, p):
    cm = pd.crosstab(p, y).values
    return cm.max(axis=1).sum() / cm.sum()


def manifold(X, seed, tag=None):
    path = os.path.join(CACHE, f"U_{tag}_s{seed}.npy") if tag else None
    if path and os.path.exists(path):
        return np.load(path)
    import umap.umap_ as umap
    Z = TruncatedSVD(fw.SVD_DIMS, random_state=seed).fit_transform(X)
    U = umap.UMAP(n_neighbors=fw.UMAP_NEIGH, n_components=fw.UMAP_DIMS, min_dist=0.0, metric="cosine",
                  random_state=seed, low_memory=True, n_epochs=fw.UMAP_EPOCHS, n_jobs=1).fit_transform(Z)
    if path:
        np.save(path, U)
    return U


def merged(y, p):
    """Describe which sub-types share a cluster (dominant share >= 25% of a cluster)."""
    ct = pd.crosstab(p, y)
    parts = []
    for c in ct.index:
        row = ct.loc[c] / ct.loc[c].sum()
        big = [k[:4] for k, v in row.items() if v >= 0.25]
        parts.append("+".join(sorted(big)))
    return " | ".join(sorted(parts))


def main():
    from rhetorical_features import clean_text_aggressive
    t00 = time.time()
    df = pd.read_csv(fw.CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    df = df.reset_index(drop=True)
    y = df["label"].to_numpy()
    F, cols = fw.load_features(df, False, [], refresh=False)
    E = fw.sbert(df["text"].map(clean_text_aggressive), f"cased_{len(df)}")
    S = normalize(np.asarray(E, dtype=np.float32))
    Rraw = MinMaxScaler().fit_transform(F[cols].to_numpy(float))
    say(f"corpus: {len(df):,} documents")
    rng = np.random.RandomState(0)
    sil_idx = rng.choice(len(df), min(5000, len(df)), replace=False)

    say("\n=== P1. 20 random Fourier draws at the reported setting (pipeline seed 42) ===")
    draws = list(range(42, 62))
    parts, rows = {}, []
    for d in draws:
        X = normalize(np.hstack([S * 1.5, fw.rff(Rraw, seed=d, D=200, g=1.0)]))
        U = manifold(X, 42, tag=f"draw{d}")
        p = KMeans(K, n_init=10, random_state=42).fit_predict(U)
        parts[d] = p
        sil = silhouette_score(U[sil_idx], p[sil_idx])
        a = adjusted_rand_score(y, p)
        rows.append({"draw": d, "ARI": round(a, 4), "NMI": round(normalized_mutual_info_score(y, p), 4),
                     "purity": round(purity(y, p), 4), "silhouette": round(sil, 4), "clusters": merged(y, p)})
        say(f"  draw {d}: ARI {a:.4f}  silhouette {sil:.4f}  clusters: {rows[-1]['clusters']}  [{(time.time()-t00)/60:.0f} min]")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "P1_draws.csv"), index=False)
    T1 = pd.DataFrame(rows)
    say(f"  ARI over 20 draws: mean {T1.ARI.mean():.4f}, sd {T1.ARI.std(ddof=1):.4f}, "
        f"min {T1.ARI.min():.4f}, max {T1.ARI.max():.4f}; share of draws with ARI >= 0.65: {(T1.ARI >= 0.65).mean():.2f}")
    say(f"  correlation silhouette vs ARI: {np.corrcoef(T1.silhouette, T1.ARI)[0, 1]:.3f}")

    say("\n=== P2. Consensus over draws (label-free cluster ensemble) ===")
    def consensus(ds, seed):
        H = np.hstack([np.eye(K)[parts[d]] for d in ds])
        return KMeans(K, n_init=20, random_state=seed).fit_predict(H)
    rows2 = []
    for name, ds in (("draws 42-51", draws[:10]), ("draws 52-61", draws[10:])):
        for s in (42, 43, 44):
            p = consensus(ds, s)
            rows2.append({"ensemble": name, "seed": s, "ARI": round(adjusted_rand_score(y, p), 4),
                          "NMI": round(normalized_mutual_info_score(y, p), 4), "purity": round(purity(y, p), 4),
                          "clusters": merged(y, p)})
            say(f"  {name}, seed {s}: ARI {rows2[-1]['ARI']:.4f}  purity {rows2[-1]['purity']:.4f}  clusters: {rows2[-1]['clusters']}")
    pA = consensus(draws[:10], 42); pB = consensus(draws[10:], 42)
    say(f"  agreement between the two independent ensembles (ARI between partitions): {adjusted_rand_score(pA, pB):.4f}")
    pd.DataFrame(rows2).to_csv(os.path.join(OUT, "P2_consensus.csv"), index=False)

    say("\n=== P3. Label-free choice of the draw by silhouette ===")
    for name, ds in (("draws 42-51", draws[:10]), ("draws 52-61", draws[10:]), ("all 20", draws)):
        sub = T1[T1.draw.isin(ds)]
        best = sub.loc[sub.silhouette.idxmax()]
        say(f"  {name}: highest silhouette = draw {int(best.draw)} -> ARI {best.ARI:.4f} (mean over these draws {sub.ARI.mean():.4f})")

    say("\n=== P4. No kernel expansion: 13 scaled features fused directly (no random draw) ===")
    grid = [(5, 1), (3, 1), (2, 1), (1.5, 1), (1, 1), (1, 1.5), (1, 2), (1, 3), (1, 5)]
    rows4 = []
    for ws, wr in grid:
        X = normalize(np.hstack([S * ws, Rraw * wr]))
        res = []
        for s in (42, 43, 44):
            U = manifold(X, s, tag=f"raw{ws:g}-{wr:g}")
            p = KMeans(K, n_init=10, random_state=s).fit_predict(U)
            res.append((adjusted_rand_score(y, p), normalized_mutual_info_score(y, p), purity(y, p),
                        silhouette_score(U[sil_idx], p[sil_idx])))
        a = np.array(res)
        rows4.append({"w": f"{ws:g}:{wr:g}", "ARI_mean": round(a[:, 0].mean(), 4), "ARI_sd": round(a[:, 0].std(ddof=1), 4),
                      "NMI_mean": round(a[:, 1].mean(), 4), "purity_mean": round(a[:, 2].mean(), 4),
                      "silhouette_mean": round(a[:, 3].mean(), 4)})
        say(f"  {ws:g}:{wr:g}  ARI {a[:, 0].mean():.4f} (sd {a[:, 0].std(ddof=1):.4f})  purity {a[:, 2].mean():.4f}  "
            f"silhouette {a[:, 3].mean():.4f}  [{(time.time()-t00)/60:.0f} min]")
        pd.DataFrame(rows4).to_csv(os.path.join(OUT, "P4_nokernel_sweep.csv"), index=False)
    T4 = pd.DataFrame(rows4)
    bsil = T4.loc[T4.silhouette_mean.idxmax()]
    say(f"  label-free choice (highest silhouette): {bsil.w} -> ARI {bsil.ARI_mean:.4f}")

    say("\n  held-out choice of the weighting for the no-kernel variant (20% select / 80% evaluate)")
    rows5 = []
    Xs = {(ws, wr): normalize(np.hstack([S * ws, Rraw * wr])) for ws, wr in grid}
    for split in (42, 43, 44, 45, 46):
        sel, ev = next(StratifiedShuffleSplit(1, train_size=0.20, random_state=split).split(df, y))
        best, ba = None, -1
        for key, X in Xs.items():
            U = manifold(X[sel], split)
            a = adjusted_rand_score(y[sel], KMeans(K, n_init=10, random_state=split).fit_predict(U))
            if a > ba:
                best, ba = key, a
        U = manifold(Xs[best][ev], split)
        p = KMeans(K, n_init=10, random_state=split).fit_predict(U)
        rows5.append({"split": split, "w": f"{best[0]:g}:{best[1]:g}", "ARI_selection": round(ba, 4),
                      "ARI_eval": round(adjusted_rand_score(y[ev], p), 4), "purity_eval": round(purity(y[ev], p), 4)})
        say(f"  split {split}: chose {rows5[-1]['w']} -> evaluation ARI {rows5[-1]['ARI_eval']:.4f}  [{(time.time()-t00)/60:.0f} min]")
        pd.DataFrame(rows5).to_csv(os.path.join(OUT, "P4_nokernel_heldout.csv"), index=False)
    T5 = pd.DataFrame(rows5)
    say(f"  held-out ARI: mean {T5.ARI_eval.mean():.4f}, sd {T5.ARI_eval.std(ddof=1):.4f}")
    say(f"\ntotal time {(time.time()-t00)/60:.0f} min")


if __name__ == "__main__":
    main()
