"""
Main results with the consensus step.

Each of M = 10 random Fourier draws (seeds 42-51) gives a K-Means partition;
the one-hot memberships are concatenated and clustered with K-Means (K = 4).

  Q1. stream-weight sweep        Q2. rhetorical stream alone
  Q3. Gaussian mixture           Q4. agreement with a single draw
  Q5. five-fold cross-validation

Output: results/consensus/
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import framework as fw  # noqa: E402
OUT = os.path.join(fw.RES, "consensus")
CACHE = os.path.join(OUT, "cache")
OLD = os.path.join(fw.RES, "draw_stability", "cache")
os.makedirs(CACHE, exist_ok=True)
K = 4
DRAWS = list(range(42, 52))
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(LOG))


def purity(y, p):
    cm = pd.crosstab(p, y).values
    return cm.max(axis=1).sum() / cm.sum()


def scores(y, p):
    return {"ARI": round(adjusted_rand_score(y, p), 4), "NMI": round(normalized_mutual_info_score(y, p), 4),
            "Purity": round(float(purity(y, p)), 4)}


def umap_model(seed):
    import umap.umap_ as umap
    return umap.UMAP(n_neighbors=fw.UMAP_NEIGH, n_components=fw.UMAP_DIMS, min_dist=0.0, metric="cosine",
                     random_state=seed, low_memory=True, n_epochs=fw.UMAP_EPOCHS, n_jobs=1)


def manifold(X, tag, seed=42):
    for d in (CACHE, OLD):
        p = os.path.join(d, f"U_{tag}_s{seed}.npy")
        if os.path.exists(p):
            return np.load(p)
    Z = TruncatedSVD(fw.SVD_DIMS, random_state=seed).fit_transform(X)
    U = umap_model(seed).fit_transform(Z)
    np.save(os.path.join(CACHE, f"U_{tag}_s{seed}.npy"), U)
    return U


def consensus(parts, seed=42):
    H = np.hstack([np.eye(K)[p] for p in parts])
    return KMeans(K, n_init=20, random_state=seed).fit_predict(H)


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
    RFF = {d: fw.rff(Rraw, seed=d, D=200, g=1.0) for d in DRAWS}
    say(f"corpus: {len(df):,} documents; consensus over draws {DRAWS[0]}-{DRAWS[-1]}")

    def fusedX(ws, wr, d):
        if ws == 0:
            return RFF[d]
        return normalize(np.hstack([S * ws, RFF[d] * wr]))

    # Q3 and Q4 reuse the manifolds cached by 08_draw_stability.py
    say("\n=== Q3/Q4. Fused 1.5:1 with consensus: K-Means and Gaussian mixture ===")
    Us = {d: manifold(fusedX(1.5, 1, d), f"draw{d}") for d in DRAWS}
    km = {d: KMeans(K, n_init=10, random_state=42).fit_predict(Us[d]) for d in DRAWS}
    for s in (42, 43, 44):
        say(f"  K-Means consensus (seed {s}): {scores(y, consensus(list(km.values()), s))}")
    pc = consensus(list(km.values()), 42)
    gm = {d: GaussianMixture(K, covariance_type="diag", n_init=5, reg_covar=1e-4, random_state=42).fit_predict(Us[d]) for d in DRAWS}
    say(f"  GMM single draws: ARI mean {np.mean([adjusted_rand_score(y, p) for p in gm.values()]):.4f}")
    for s in (42, 43, 44):
        say(f"  GMM consensus (seed {s}): {scores(y, consensus(list(gm.values()), s))}")
    say(f"  agreement consensus vs single draw 42 (ARI between partitions): {adjusted_rand_score(pc, km[42]):.4f}")
    ct = pd.crosstab(pc, y)
    ct.to_csv(os.path.join(OUT, "Q3_consensus_confusion.csv"))
    say("  consensus confusion (rows = clusters):\n" + ct.to_string())

    # Q2 rhetorical alone
    say("\n=== Q2. Rhetorical stream alone (expanded), with consensus ===")
    parts = []
    for d in DRAWS:
        U = manifold(fusedX(0, 1, d), f"rhet_draw{d}")
        parts.append(KMeans(K, n_init=10, random_state=42).fit_predict(U))
    say(f"  single draws: ARI mean {np.mean([adjusted_rand_score(y, p) for p in parts]):.4f}; "
        f"consensus: {scores(y, consensus(parts))}")

    # Q1 weight sweep
    say("\n=== Q1. Weight sweep with consensus (draws 42-51, pipeline seed 42) ===")
    rows = []
    for ws, wr in [(5, 1), (4, 1), (3, 1), (2, 1), (1.5, 1), (1, 1), (1, 1.5), (1, 2), (1, 3)]:
        parts = []
        for d in DRAWS:
            tag = f"draw{d}" if (ws, wr) == (1.5, 1) else f"w{ws:g}-{wr:g}_draw{d}"
            U = manifold(fusedX(ws, wr, d), tag)
            parts.append(KMeans(K, n_init=10, random_state=42).fit_predict(U))
        single = [adjusted_rand_score(y, p) for p in parts]
        sc = scores(y, consensus(parts))
        rows.append({"w": f"{ws:g}:{wr:g}", "single_draw_ARI_mean": round(np.mean(single), 4),
                     "single_draw_ARI_min": round(np.min(single), 4), "single_draw_ARI_max": round(np.max(single), 4),
                     **{f"consensus_{k}": v for k, v in sc.items()}})
        say(f"  {ws:g}:{wr:g}  single draws {np.mean(single):.4f} [{np.min(single):.4f}-{np.max(single):.4f}]  "
            f"consensus ARI {sc['ARI']:.4f}  [{(time.time()-t00)/60:.0f} min]")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "Q1_weight_sweep_consensus.csv"), index=False)

    # Q5 cross-validation
    say("\n=== Q5. Five-fold cross-validation with consensus ===")
    rows5 = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(skf.split(df, y), 1):
        parts_te, parts_tr = [], []
        for d in DRAWS:
            X = fusedX(1.5, 1, d)
            svd = TruncatedSVD(fw.SVD_DIMS, random_state=42).fit(X[tr])
            um = umap_model(42).fit(svd.transform(X[tr]))
            Utr = um.embedding_
            Ute = um.transform(svd.transform(X[te]))
            kmm = KMeans(K, n_init=10, random_state=42).fit(Utr)
            parts_tr.append(kmm.labels_); parts_te.append(kmm.predict(Ute))
        a_tr = adjusted_rand_score(y[tr], consensus(parts_tr))
        a_te = adjusted_rand_score(y[te], consensus(parts_te))
        rows5.append({"fold": fold, "ARI_fitted": round(a_tr, 4), "ARI_heldout": round(a_te, 4)})
        say(f"  fold {fold}: fitted {a_tr:.4f}  held-out {a_te:.4f}  [{(time.time()-t00)/60:.0f} min]")
        pd.DataFrame(rows5).to_csv(os.path.join(OUT, "Q5_cv_consensus.csv"), index=False)
    T = pd.DataFrame(rows5)
    say(f"  held-out ARI mean {T.ARI_heldout.mean():.4f} (sd {T.ARI_heldout.std(ddof=1):.4f}); "
        f"fitted {T.ARI_fitted.mean():.4f}")
    say(f"\ntotal time {(time.time()-t00)/60:.0f} min")


if __name__ == "__main__":
    main()
