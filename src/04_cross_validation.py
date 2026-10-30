"""
Held-out evaluation.

    --part folds        five-fold cross-validation (fit on four folds, apply to the fifth)
    --part collections  fit on one collection, apply to the other
    --part external     apply to the external labelled set (data/external/)
    --part supervised   supervised classifier on the same features (upper reference)
    --part all

Output: results/cv_*.csv
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
XD = os.path.join(DATA, "external")
sys.path.insert(0, HERE)


import framework as fw

RS = 42
WS, WR = 1.5, 1.0
K = 4
SVD_DIMS, UMAP_NEIGH, UMAP_DIMS, UMAP_EPOCHS = 100, 15, 40, 200


#  a pipeline that can be fitted and then applied to unseen documents
class Pipeline:
    """Fit on one set of documents, apply to another."""

    def __init__(self, ws=WS, wr=WR, k=K, algo="kmeans", seed=RS,
                 use_rbf=True):
        self.ws, self.wr, self.k = ws, wr, k
        self.algo, self.seed, self.use_rbf = algo, seed, use_rbf

    def _stream(self, S, Fraw, fit=False):
        if fit:
            self.scaler = MinMaxScaler().fit(Fraw)
        R = self.scaler.transform(Fraw)
        R = np.clip(R, 0, 1)          # unseen documents can fall outside
        R = fw.rff(R) if self.use_rbf else normalize(R)
        return normalize(np.hstack([S * self.ws, R.astype(np.float32)
                                    * self.wr]))

    def fit(self, S, Fraw):
        import umap.umap_ as umap
        X = self._stream(S, Fraw, fit=True)
        self.svd = TruncatedSVD(min(SVD_DIMS, X.shape[1] - 1),
                                random_state=self.seed).fit(X)
        Z = self.svd.transform(X)
        self.umap = umap.UMAP(n_neighbors=UMAP_NEIGH,
                              n_components=UMAP_DIMS, min_dist=0.0,
                              metric="cosine", random_state=self.seed,
                              low_memory=True, n_epochs=UMAP_EPOCHS,
                              n_jobs=1).fit(Z)
        U = self.umap.embedding_
        if self.algo == "gmm":
            self.model = GaussianMixture(self.k, covariance_type="diag",
                                         n_init=5, reg_covar=1e-4,
                                         random_state=self.seed).fit(U)
        else:
            self.model = KMeans(self.k, n_init=10,
                                random_state=self.seed).fit(U)
        self.train_U = U
        return self

    def predict(self, S, Fraw):
        X = self._stream(S, Fraw, fit=False)
        U = self.umap.transform(self.svd.transform(X))
        return self.model.predict(U)

    def predict_train(self):
        return self.model.predict(self.train_U)


def load_main():
    df = pd.read_csv(fw.CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    cache = os.path.join(DATA, f"sbert_cased_{len(df)}.npy")
    if not os.path.exists(cache):
        raise SystemExit(f"missing {cache} - run framework.py first")
    S = normalize(np.load(cache).astype(np.float32))
    F, pub = fw.load_features(df, False, [])
    return df, S, F[pub].to_numpy(float), pub


def score_split(y, pred, classes, **extra):
    return fw.score(np.asarray(y), np.asarray(pred), classes, **extra)


def part_folds(df, S, Fraw, classes, a):
    out = os.path.join(RES, "cv_folds.csv")
    y = df["label"].to_numpy()
    rows = []
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=RS)
    for i, (tr, te) in enumerate(skf.split(S, y), start=1):
        for algo in ("kmeans", "gmm"):
            t0 = time.time()
            pl = Pipeline(algo=algo).fit(S[tr], Fraw[tr])
            pred_te = pl.predict(S[te], Fraw[te])
            r_te = score_split(y[te], pred_te, classes,
                               fold=i, algorithm=algo, split="held-out",
                               n=len(te))
            r_tr = score_split(y[tr], pl.predict_train(), classes,
                               fold=i, algorithm=algo, split="training",
                               n=len(tr))
            rows += [r_te, r_tr]
            print(f"  fold {i} {algo:<8} held-out ARI {r_te['ARI']:.4f}   "
                  f"training ARI {r_tr['ARI']:.4f}   "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            del pl
            gc.collect()
    T = pd.DataFrame(rows)
    T.to_csv(out, index=False)
    print("\n" + "=" * 74)
    print(f"  {a.folds}-FOLD CROSS-VALIDATION")
    print("=" * 74)
    print(T.groupby(["algorithm", "split"])[
        ["ARI", "NMI", "Purity", "Match %"]].agg(["mean", "std"])
        .round(4).to_string())
    for algo in T.algorithm.unique():
        s = T[T.algorithm == algo]
        gap = (s[s.split == "training"].ARI.mean()
               - s[s.split == "held-out"].ARI.mean())
        print(f"\n  {algo}: held-out is {gap:+.4f} ARI below training")
        if gap > 0.10:
            print("    A gap this size means the embedding is arranging the")
            print("    documents it was fitted to more than it is recovering")
            print("    structure that generalises. Report it.")
    print(f"\nwritten: {out}")


def part_collections(df, S, Fraw, classes, a):
    out = os.path.join(RES, "cv_collections.csv")
    y = df["label"].to_numpy()
    coll = df["collection"].to_numpy()
    rows = []
    for train_c, test_c in [(1, 2), (2, 1)]:
        tr = np.where(coll == train_c)[0]
        te = np.where(coll == test_c)[0]
        for algo in ("kmeans", "gmm"):
            t0 = time.time()
            pl = Pipeline(algo=algo).fit(S[tr], Fraw[tr])
            rows.append(score_split(
                y[te], pl.predict(S[te], Fraw[te]), classes,
                fitted_on=f"collection {train_c}",
                applied_to=f"collection {test_c}", algorithm=algo,
                n=len(te)))
            rows.append(score_split(
                y[tr], pl.predict_train(), classes,
                fitted_on=f"collection {train_c}",
                applied_to=f"collection {train_c}", algorithm=algo,
                n=len(tr)))
            print(f"  fit on {train_c} -> apply to {test_c}  {algo:<8}"
                  f"ARI {rows[-2]['ARI']:.4f}   (in-sample "
                  f"{rows[-1]['ARI']:.4f})   [{time.time()-t0:.0f}s]",
                  flush=True)
            del pl
            gc.collect()
    T = pd.DataFrame(rows)
    T.to_csv(out, index=False)
    print("\n" + "=" * 74)
    print("  TRANSFER BETWEEN THE TWO COLLECTIONS")
    print("=" * 74)
    print(T[["fitted_on", "applied_to", "algorithm", "n", "ARI", "NMI",
             "Purity", "Match %"]].to_string(index=False))
    print(f"\nwritten: {out}")


def part_external(df, S, Fraw, classes, a):
    """Apply the fitted pipeline to a separately labelled corpus."""
    out = os.path.join(RES, "cv_external.csv")
    pool = os.path.join(XD, "eval_balanced_pool.csv")
    l1 = os.path.join(XD, "labels_local.csv")
    l2 = os.path.join(XD, "labels_qwen25.csv")
    for p in (pool, l1, l2):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    P = pd.read_csv(pool)
    A = pd.read_csv(l1)[["doc_id", "ai_label"]].rename(
        columns={"ai_label": "label_a"})
    B = pd.read_csv(l2)[["doc_id", "ai_label"]].rename(
        columns={"ai_label": "label_b"})
    M = P.merge(A, on="doc_id").merge(B, on="doc_id")
    text_col = "text_full" if "text_full" in M.columns else "text"
    M = M[M[text_col].astype(str).str.split().str.len() >= 10].reset_index(
        drop=True)

    both = M[M.label_a.isin(classes) & M.label_b.isin(classes)]
    kappa = float(cohen_kappa_score(both.label_a, both.label_b))
    agreed = both[both.label_a == both.label_b]

    print(f"  {len(M)} labelled documents")
    print(f"  {len(both)} have both labels among the four classes; "
          f"Cohen's kappa {kappa:.4f}")
    print(f"  {len(agreed)} of those agree, {len(both)-len(agreed)} disagree")
    print(f"  agreed composition: {dict(agreed.label_a.value_counts())}")
    print("  agreement is a proxy for ease, so the agreed subset is the "
          "easiest\n  documents in the pool. All three targets are reported "
          "below.\n")

    from rhetorical_features import clean_text_aggressive
    from rhetorical_features import extract_rhetorical_features
    texts = M[text_col].astype(str).tolist()
    cache = os.path.join(DATA, f"sbert_external_all_{len(texts)}.npy")
    if os.path.exists(cache):
        E = np.load(cache)
        print(f"  SBERT: cached {E.shape}")
    else:
        from sentence_transformers import SentenceTransformer
        E = SentenceTransformer("all-MiniLM-L6-v2").encode(
            [clean_text_aggressive(t) for t in texts],
            show_progress_bar=True, batch_size=64)
        np.save(cache, E)
    Se_all = normalize(np.asarray(E, dtype=np.float32))
    d = extract_rhetorical_features(pd.DataFrame({"text": texts}))
    Fe_all = d[[c for c in d.columns if c.startswith("f_")]].to_numpy(float)

    rows = []
    for algo in ("kmeans", "gmm"):
        pl = Pipeline(algo=algo).fit(S, Fraw)
        pred_all = pl.predict(Se_all, Fe_all)          # one pass, reused
        del pl
        gc.collect()

        def sc(mask, y, **kw):
            yy = np.asarray(y)
            pp = pred_all[mask]
            cc = [c for c in classes if (yy == c).any()]
            return score_split(yy, pp, cc, algorithm=algo,
                               kappa_between_labellers=round(kappa, 4), **kw)

        m = M.index.isin(agreed.index)
        r = sc(m, agreed.label_a.to_numpy(), target="agreed subset",
               n=int(m.sum()), draws=1)
        rows.append(r)
        print(f"  {algo:<8} agreed subset      n {r['n']:>5}   "
              f"ARI {r['ARI']:.4f}   purity {r['Purity']:.4f}", flush=True)

        per = int(agreed.label_a.value_counts().min())
        draws = []
        for s in range(a.draws):
            idx = np.concatenate([
                g.sample(per, random_state=RS + s).index.to_numpy()
                for _, g in agreed.groupby("label_a")])
            mm = M.index.isin(idx)
            sub = M.loc[mm]
            draws.append(sc(mm, agreed.loc[sub.index].label_a.to_numpy(),
                            target="balanced", n=int(mm.sum()), draw=s))
        D = pd.DataFrame(draws)
        r = dict(D.mean(numeric_only=True))
        r.update(algorithm=algo, target=f"balanced ({per} per class)",
                 n=int(D.n.iloc[0]), draws=a.draws,
                 ARI_sd=round(float(D.ARI.std()), 4),
                 kappa_between_labellers=round(kappa, 4))
        for k in ("ARI", "NMI", "Purity", "Match %", "Homogeneity"):
            if k in r:
                r[k] = round(float(r[k]), 4)
        rows.append(r)
        print(f"  {algo:<8} balanced           n {r['n']:>5}   "
              f"ARI {r['ARI']:.4f} +/- {r['ARI_sd']:.4f}   "
              f"purity {r['Purity']:.4f}   ({a.draws} draws, {per} per class)",
              flush=True)

        for name, col in (("labeller A (llama)", "label_a"),
                          ("labeller B (qwen)", "label_b")):
            sub = M[M[col].isin(classes)]
            mm = M.index.isin(sub.index)
            r = sc(mm, sub[col].to_numpy(), target=name, n=int(mm.sum()),
                   draws=1)
            rows.append(r)
            print(f"  {algo:<8} {name:<18} n {r['n']:>5}   "
                  f"ARI {r['ARI']:.4f}   purity {r['Purity']:.4f}",
                  flush=True)

    T = pd.DataFrame(rows)
    cols = ["target", "algorithm", "n", "ARI", "ARI_sd", "NMI", "Purity",
            "Match %", "kappa_between_labellers"]
    T = T[[c for c in cols if c in T.columns]]
    T.to_csv(out, index=False)
    print("\n" + "=" * 78)
    print("  EXTERNAL VALIDATION - fitted on the study corpus, applied "
          "elsewhere")
    print("=" * 78)
    print(T.to_string(index=False))
    print("\n  The agreed subset is the most favourable of the three and "
          "should be\n  read as an upper bound. The balanced rows remove the "
          "class imbalance;\n  the per-labeller rows remove the agreement "
          "filter entirely.")
    print(f"\n  Cohen's kappa between the two labellers is {kappa:.4f}. No "
          f"method can\n  agree with a target more reliably than the target "
          f"agrees with itself.")
    print(f"\nwritten: {out}")


def part_supervised(df, S, Fraw, classes, a):
    out = os.path.join(RES, "cv_supervised.csv")
    y = df["label"].to_numpy()
    rows = []
    sets = {
        "rhetorical features (13)": Fraw,
        "document length (1)": df["text"].str.split().str.len()
        .to_numpy().reshape(-1, 1),
        "semantic embedding (384)": S,
        "both streams (395)": np.hstack([S, Fraw]),
    }
    for name, X in sets.items():
        t0 = time.time()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
        s = cross_val_score(HistGradientBoostingClassifier(random_state=RS),
                            X, y, cv=cv, scoring="accuracy")
        rows.append({"features": name, "dim": X.shape[1],
                     "accuracy": round(100 * s.mean(), 2),
                     "sd": round(100 * s.std(), 2),
                     "seconds": round(time.time() - t0)})
        print(f"  {name:<28} {100*s.mean():6.2f}% +/- {100*s.std():.2f}",
              flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(out, index=False)
    print("\n" + "=" * 74)
    print("  SUPERVISED CEILING - five-fold cross-validation")
    print("=" * 74)
    print(T.to_string(index=False))
    print(f"\n  chance is {100/len(classes):.2f}%. These are ceilings, not")
    print("  competitors: they say what is recoverable when labels are seen.")
    print(f"\nwritten: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["folds", "collections", "external",
                             "supervised", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=10,
                    help="balanced subsamples of the external set to average")
    a = ap.parse_args()

    df, S, Fraw, pub = load_main()
    classes = [c for c in fw.CLASSES if (df.label == c).any()]
    print(f"{len(df):,} documents, {len(pub)} rhetorical features, "
          f"weights {WS:g}:{WR:g}, K = {K}\n")

    parts = ([a.part] if a.part != "all"
             else ["supervised", "folds", "collections", "external"])
    for p in parts:
        print("=" * 74)
        print(f"  PART: {p}")
        print("=" * 74)
        {"folds": part_folds, "collections": part_collections,
         "external": part_external, "supervised": part_supervised}[p](
            df, S, Fraw, classes, a)
        print()


if __name__ == "__main__":
    main()
