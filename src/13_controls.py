"""
Source and length controls.

  A. Corpus without the hyperpartisan source (SemEval-2019 Task 4).
  B. Length-matched samples of the four sub-types (2-word length bins, five samples).
  C. Length-matched samples of rumor, satire and clickbait (K = 3).

Each control refits the whole pipeline, including the consensus over ten random draws,
on the documents retained, and reruns the embedding, artifact and length baselines.

Output: results/controls/
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
OUT = os.path.join(fw.RES, "controls")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
DRAWS = list(range(42, 52))
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(LOG))


def scores(y, p):
    cm = pd.crosstab(p, y).values
    r, c = linear_sum_assignment(-cm)
    return {"ARI": round(adjusted_rand_score(y, p), 4), "NMI": round(normalized_mutual_info_score(y, p), 4),
            "Purity": round(float(cm.max(axis=1).sum() / cm.sum()), 4), "Match": round(float(100 * cm[r, c].sum() / cm.sum()), 2)}


def manifold(X, tag, seed=42):
    path = os.path.join(CACHE, f"U_{tag}_s{seed}.npy") if tag else None
    if path and os.path.exists(path):
        return np.load(path)
    if X.shape[1] <= 2:
        U = X
    else:
        import umap.umap_ as umap
        Z = TruncatedSVD(min(fw.SVD_DIMS, X.shape[1] - 1), random_state=seed).fit_transform(X)
        U = umap.UMAP(n_neighbors=fw.UMAP_NEIGH, n_components=min(fw.UMAP_DIMS, Z.shape[1] - 1), min_dist=0.0,
                      metric="cosine", random_state=seed, low_memory=True, n_epochs=fw.UMAP_EPOCHS,
                      n_jobs=1).fit_transform(Z)
    if path:
        np.save(path, U)
    return U


def consensus(parts, k, seed=42):
    H = np.hstack([np.eye(k)[p] for p in parts])
    return KMeans(k, n_init=20, random_state=seed).fit_predict(H)


def run_set(name, idx, df, S_all, F_all, k, tag, base_seeds=(42, 43, 44)):
    """All conditions on the documents idx. Returns a dict condition -> scores."""
    y = df["label"].to_numpy()[idx]
    S = S_all[idx]
    Rraw = MinMaxScaler().fit_transform(F_all[idx])
    texts = df["text"].iloc[idx]
    ART = normalize(MinMaxScaler().fit_transform(fw.artefact_features(texts)))
    LEN = MinMaxScaler().fit_transform(texts.str.split().str.len().to_numpy().reshape(-1, 1))
    res = {}
    parts = []
    for d in DRAWS:
        X = normalize(np.hstack([S * 1.5, fw.rff(Rraw, seed=d, D=200, g=1.0)]))
        U = manifold(X, f"{tag}_fused_d{d}" if tag else None)
        parts.append(KMeans(k, n_init=10, random_state=42).fit_predict(U))
    res["fused_consensus"] = scores(y, consensus(parts, k))
    res["fused_single_draw_mean_ARI"] = round(float(np.mean([adjusted_rand_score(y, p) for p in parts])), 4)
    for cname, X in (("semantic", S), ("artifact", ART), ("length", LEN)):
        sc = []
        for s in base_seeds:
            U = manifold(X, f"{tag}_{cname}" if tag else None, s)
            sc.append(scores(y, KMeans(k, n_init=10, random_state=s).fit_predict(U)))
        res[cname] = {m: round(float(np.mean([x[m] for x in sc])), 4) for m in sc[0]}
    return res, y


def main():
    from rhetorical_features import clean_text_aggressive
    t00 = time.time()
    df = pd.read_csv(fw.CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    df = df.reset_index(drop=True)
    F, cols = fw.load_features(df, False, [], refresh=False)
    F_all = F[cols].to_numpy(float)
    E = fw.sbert(df["text"].map(clean_text_aggressive), f"cased_{len(df)}")
    S_all = normalize(np.asarray(E, dtype=np.float32))
    nw = df["text"].str.split().str.len().to_numpy()
    say(f"corpus: {len(df):,} documents")
    rows = []

    # ---------------- A
    say("\n=== A. Without the hyperpartisan source ===")
    idx = np.where(df["src"].to_numpy() != "SemEval2019-hyperpartisan")[0]
    say(f"  documents: {len(idx):,}  " + str(df["label"].iloc[idx].value_counts().to_dict()))
    res, y = run_set("A", idx, df, S_all, F_all, 4, "nohyp")
    for c, v in res.items():
        say(f"  {c:<28} {v}")
        if isinstance(v, dict):
            rows.append({"control": "A_no_hyperpartisan", "sample": 0, "condition": c, **v})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "controls.csv"), index=False)
    say(f"  [{(time.time()-t00)/60:.0f} min]")

    # ---------------- B and C
    for code, classes in (("B", ["propaganda", "rumour", "satire", "clickbait"]), ("C", ["rumour", "satire", "clickbait"])):
        say(f"\n=== {code}. Length-matched sample, {len(classes)} sub-types (2-word bins, five samples) ===")
        k = len(classes)
        keep = []
        for smp in (42, 43, 44, 45, 46):
            rng = np.random.RandomState(smp)
            sel = []
            pool = df[df["label"].isin(classes)].assign(b=nw[df["label"].isin(classes)] // 2)
            for b, g in pool.groupby("b"):
                counts = g["label"].value_counts()
                if not all(c in counts for c in classes):
                    continue
                n = counts[classes].min()
                for c in classes:
                    ids = g.index[g["label"] == c].to_numpy()
                    sel.extend(rng.choice(ids, n, replace=False))
            idx = np.array(sorted(sel))
            res, y = run_set(code, idx, df, S_all, F_all, k, None, base_seeds=(42,))
            mean_len = pd.Series(nw[idx]).groupby(y).mean().round(1).to_dict()
            say(f"  sample {smp}: {len(idx)} documents, mean length by sub-type {mean_len}")
            for c, v in res.items():
                if isinstance(v, dict):
                    say(f"    {c:<26} ARI {v['ARI']:.4f}  NMI {v['NMI']:.4f}  match {v['Match']:.2f}")
                    rows.append({"control": f"{code}_length_matched_{k}class", "sample": smp, "condition": c, **v})
            keep.append(res)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "controls.csv"), index=False)
        T = pd.DataFrame([r for r in rows if r["control"].startswith(code)])
        say(f"  mean over samples (sd):")
        for c, g in T.groupby("condition", sort=False):
            say(f"    {c:<26} ARI {g.ARI.mean():.4f} ({g.ARI.std(ddof=1):.4f})  NMI {g.NMI.mean():.4f}  match {g.Match.mean():.2f}")
        say(f"  [{(time.time()-t00)/60:.0f} min]")
    say(f"\ntotal time {(time.time()-t00)/60:.0f} min")


if __name__ == "__main__":
    main()
