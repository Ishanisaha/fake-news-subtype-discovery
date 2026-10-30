"""
Held-out choice of the stream weighting.

For five stratified splits, the weighting is chosen on 20% of the documents and
the chosen weighting is evaluated on the other 80%.

Output: results/weight_holdout/
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import framework as fw

GRID = [(1, 0), (5, 1), (4, 1), (3, 1), (2, 1), (1.5, 1), (1, 1), (1, 1.5), (1, 2), (1, 3), (0, 1)]
SPLITS = [42, 43, 44, 45, 46]
SEL_SHARE = 0.20
K = 4
OUT = os.path.join(fw.RES, "weight_holdout")
os.makedirs(OUT, exist_ok=True)
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def manifold(X, seed):
    import umap.umap_ as umap
    Z = TruncatedSVD(min(fw.SVD_DIMS, X.shape[1] - 1), random_state=seed).fit_transform(X)
    return umap.UMAP(n_neighbors=fw.UMAP_NEIGH, n_components=fw.UMAP_DIMS, min_dist=0.0, metric="cosine",
                     random_state=seed, low_memory=True, n_epochs=fw.UMAP_EPOCHS, n_jobs=1).fit_transform(Z)


def run(X, y, seed):
    U = manifold(X, seed)
    p = KMeans(K, n_init=10, random_state=seed).fit_predict(U)
    return adjusted_rand_score(y, p), normalized_mutual_info_score(y, p)


def main():
    from rhetorical_features import clean_text_aggressive
    t00 = time.time()
    df = pd.read_csv(fw.CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    y = df["label"].to_numpy()
    classes = [c for c in fw.CLASSES if (y == c).any()]
    if df["text"].str.contains(r"\[SEP\]").any():
        sys.exit("The corpus still contains [SEP]. Run 01_clean_corpus.py first.")
    F, cols = fw.load_features(df, False, [], refresh=False)
    say(f"corpus: {len(df):,} documents, rhetorical features: {len(cols)}")
    E = fw.sbert(df["text"].map(clean_text_aggressive), f"cased_{len(df)}")
    S = normalize(np.asarray(E, dtype=np.float32))
    R = fw.rff(MinMaxScaler().fit_transform(F[cols].to_numpy(float)))   # label-free scaling

    def fused(ws, wr):
        if wr == 0:
            return S
        if ws == 0:
            return R
        return normalize(np.hstack([S * ws, R * wr]))

    rows, sweeps = [], []
    for seed in SPLITS:
        sel, ev = next(StratifiedShuffleSplit(1, train_size=SEL_SHARE, random_state=seed).split(df, y))
        say(f"\nsplit {seed}: selection {len(sel):,} documents, evaluation {len(ev):,} documents")
        best, best_ari = None, -1
        for ws, wr in GRID:
            a, _ = run(fused(ws, wr)[sel], y[sel], seed)
            sweeps.append({"split": seed, "w": f"{ws:g}:{wr:g}", "ARI_selection": round(a, 4)})
            say(f"  selection  {ws:g}:{wr:g}   ARI {a:.4f}")
            if a > best_ari:
                best, best_ari = (ws, wr), a
        rec = {"split": seed, "selected": f"{best[0]:g}:{best[1]:g}", "ARI_selection_best": round(best_ari, 4)}
        U = manifold(fused(*best)[ev], seed)
        for alg, p in (("KMeans", KMeans(K, n_init=10, random_state=seed).fit_predict(U)), ("GMM", fw.gmm(U, K, seed))):
            for m, v in fw.score(y[ev], p, classes).items():
                rec[f"{alg}_{m}"] = v
        if best != (1.5, 1):
            a15, _ = run(fused(1.5, 1)[ev], y[ev], seed)
            rec["KMeans_ARI_if_1.5:1"] = round(a15, 4)
        else:
            rec["KMeans_ARI_if_1.5:1"] = rec["KMeans_ARI"]
        rows.append(rec)
        say(f"  -> selected {rec['selected']}   evaluation: K-Means ARI {rec['KMeans_ARI']:.4f}, "
            f"GMM ARI {rec['GMM_ARI']:.4f}   [{(time.time()-t00)/60:.0f} min]")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_split.csv"), index=False)
        pd.DataFrame(sweeps).to_csv(os.path.join(OUT, "selection_sweeps.csv"), index=False)

    T = pd.DataFrame(rows)
    say("\n================ SUMMARY ================")
    say(T.to_string(index=False))
    say(f"\nselected weighting: {T.selected.value_counts().to_dict()}")
    for c in ["KMeans_ARI", "KMeans_NMI", "KMeans_Purity", "KMeans_Match %", "GMM_ARI", "GMM_NMI", "GMM_Purity",
              "GMM_Match %", "KMeans_ARI_if_1.5:1"]:
        say(f"evaluation part, {c:<22} mean {T[c].mean():.4f}  sd {T[c].std():.4f}")
    say(f"total time {(time.time()-t00)/60:.0f} min")
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(LOG))


if __name__ == "__main__":
    main()
