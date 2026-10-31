"""
Robustness to the sentence encoder: the main result repeated with all-mpnet-base-v2 (768 dimensions)
instead of all-MiniLM-L6-v2. Embedding alone (seeds 42-44) and fused consensus at 2:1, 1.5:1 and 1:1.

Output: results/encoder/
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import framework as fw  # noqa: E402
OUT = os.path.join(fw.RES, "encoder")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
MODEL = "all-mpnet-base-v2"
DRAWS = list(range(42, 52))
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(LOG))


def scores(y, p):
    cm = pd.crosstab(p, y).values
    r, c = linear_sum_assignment(-cm)
    return {"ARI": round(float(adjusted_rand_score(y, p)), 4), "NMI": round(float(normalized_mutual_info_score(y, p)), 4),
            "Purity": round(float(cm.max(axis=1).sum() / cm.sum()), 4), "Match": round(float(100 * cm[r, c].sum() / cm.sum()), 2)}


def manifold(X, tag, seed=42):
    path = os.path.join(CACHE, f"U_{tag}_s{seed}.npy")
    if os.path.exists(path):
        return np.load(path)
    import umap.umap_ as umap
    Z = TruncatedSVD(fw.SVD_DIMS, random_state=seed).fit_transform(X)
    U = umap.UMAP(n_neighbors=fw.UMAP_NEIGH, n_components=fw.UMAP_DIMS, min_dist=0.0, metric="cosine",
                  random_state=seed, low_memory=True, n_epochs=fw.UMAP_EPOCHS, n_jobs=1).fit_transform(Z)
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
    F, cols = fw.load_features(df, False, [], refresh=False)
    Rraw = MinMaxScaler().fit_transform(F[cols].to_numpy(float))

    emb = os.path.join(CACHE, f"sbert_mpnet_{len(df)}.npy")
    if os.path.exists(emb):
        E = np.load(emb)
    else:
        from sentence_transformers import SentenceTransformer
        say(f"encoding {len(df):,} documents with {MODEL} (first run only)")
        E = SentenceTransformer(MODEL).encode(list(df["text"].map(clean_text_aggressive)), show_progress_bar=True, batch_size=32)
        np.save(emb, E)
    S = normalize(np.asarray(E, dtype=np.float32))
    say(f"corpus: {len(df):,} documents; encoder {MODEL}, {S.shape[1]} dimensions  [{(time.time()-t00)/60:.0f} min]")
    rows = []

    say("\n=== A. Embedding alone ===")
    sc = []
    for s in (42, 43, 44):
        U = manifold(S, "mpnet_semantic", s)
        sc.append(scores(y, KMeans(4, n_init=10, random_state=s).fit_predict(U)))
    m = {k: round(float(np.mean([x[k] for x in sc])), 4) for k in sc[0]}
    say(f"  {m}")
    rows.append({"condition": "semantic_only", **m})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "encoder_check.csv"), index=False)

    say("\n=== B. Fused consensus ===")
    for ws, wr in ((2, 1), (1.5, 1), (1, 1)):
        parts = []
        for d in DRAWS:
            X = normalize(np.hstack([S * ws, fw.rff(Rraw, seed=d, D=200, g=1.0) * wr]))
            U = manifold(X, f"mpnet_w{ws:g}-{wr:g}_d{d}")
            parts.append(KMeans(4, n_init=10, random_state=42).fit_predict(U))
        res = []
        for s in (42, 43, 44):
            H = np.hstack([np.eye(4)[p] for p in parts])
            res.append(scores(y, KMeans(4, n_init=20, random_state=s).fit_predict(H)))
        m = {k: round(float(np.mean([x[k] for x in res])), 4) for k in res[0]}
        single = np.mean([adjusted_rand_score(y, p) for p in parts])
        say(f"  {ws:g}:{wr:g}  consensus {m}  single-draw mean ARI {single:.4f}  [{(time.time()-t00)/60:.0f} min]")
        rows.append({"condition": f"fused_{ws:g}-{wr:g}", **m, "single_draw_mean_ARI": round(float(single), 4)})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "encoder_check.csv"), index=False)
    say(f"\ntotal time {(time.time()-t00)/60:.0f} min")


if __name__ == "__main__":
    main()
