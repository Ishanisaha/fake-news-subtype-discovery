"""
Comparison with unsupervised text-clustering methods on the same corpus and
evaluation protocol: TF-IDF + K-Means, LDA, SBERT + K-Means,
SBERT + UMAP + K-Means and BERTopic.

    python 05_baselines.py
    python 05_baselines.py --only lda
    python 05_baselines.py --list

Output: results/sota_comparison.csv
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
sys.path.insert(0, HERE)


import framework as fw

RS = 42
WS, WR, K = 1.5, 1.0, 4
OUT = os.path.join(RES, "sota_comparison.csv")

METHODS = ["tfidf_kmeans", "lda", "sbert_kmeans", "sbert_umap_kmeans",
           "bertopic", "proposed", "artefact", "length", "supervised"]

SLOW = {"sbert_umap_kmeans", "bertopic", "proposed"}   # these fit a manifold


def load():
    df = pd.read_csv(fw.CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    cache = os.path.join(DATA, f"sbert_cased_{len(df)}.npy")
    S = (normalize(np.load(cache).astype(np.float32))
         if os.path.exists(cache) else None)
    return df, S


def clean_for_bow(texts):
    """Lower-cased text without URLs, for the bag-of-words methods. Those
    methods have no use for capitalisation and are penalised by it."""
    from rhetorical_features import clean_text_aggressive
    return [clean_text_aggressive(t) for t in texts]


def run_method(name, df, S, a):
    y = df["label"].to_numpy()
    classes = [c for c in fw.CLASSES if (y == c).any()]
    info = ""
    t0 = time.time()

    if name == "tfidf_kmeans":
        info = "surface word counts"
        X = TfidfVectorizer(max_features=20000, min_df=5,
                            stop_words="english").fit_transform(
                                clean_for_bow(df["text"]))
        Z = TruncatedSVD(100, random_state=RS).fit_transform(X)
        pred = KMeans(K, n_init=10, random_state=RS).fit_predict(normalize(Z))

    elif name == "lda":
        info = "topic proportions"
        X = CountVectorizer(max_features=20000, min_df=5,
                            stop_words="english").fit_transform(
                                clean_for_bow(df["text"]))
        lda = LatentDirichletAllocation(n_components=K, learning_method="online",
                                        max_iter=10, random_state=RS,
                                        n_jobs=1).fit(X)
        pred = lda.transform(X).argmax(axis=1)

    elif name == "sbert_kmeans":
        info = "sentence embeddings"
        if S is None:
            raise SystemExit("SBERT cache missing - run framework.py")
        pred = KMeans(K, n_init=10, random_state=RS).fit_predict(S)

    elif name == "sbert_umap_kmeans":
        info = "sentence embeddings, manifold"
        U = fw.umap40(S)
        pred = KMeans(K, n_init=10, random_state=RS).fit_predict(U)
        del U

    elif name == "bertopic":
        info = "embeddings, manifold, density"
        try:
            from bertopic import BERTopic
        except ImportError:
            print("  bertopic not installed:  pip install bertopic")
            return None
        from sklearn.cluster import KMeans as KM
        import umap.umap_ as umap
        topic_model = BERTopic(
            umap_model=umap.UMAP(n_neighbors=15, n_components=40,
                                 min_dist=0.0, metric="cosine",
                                 random_state=RS),
            hdbscan_model=KM(K, n_init=10, random_state=RS),
            calculate_probabilities=False, verbose=False)
        topics, _ = topic_model.fit_transform(clean_for_bow(df["text"]),
                                              embeddings=S)
        pred = np.asarray(topics)

    elif name == "proposed":
        info = "embeddings + rhetorical features"
        F, pub = fw.load_features(df, False, [])
        R = fw.rff(MinMaxScaler().fit_transform(
            F[pub].to_numpy(float))).astype(np.float32)
        X = normalize(np.hstack([S * WS, R * WR]))
        U = fw.umap40(X)
        pred = KMeans(K, n_init=10, random_state=RS).fit_predict(U)
        del X, U, R

    elif name == "artefact":
        info = "no linguistic content"
        A = normalize(MinMaxScaler().fit_transform(
            fw.artefact_features(df["text"])))
        U = fw.umap40(A)
        pred = KMeans(K, n_init=10, random_state=RS).fit_predict(U)
        del U

    elif name == "length":
        info = "document length only"
        L = MinMaxScaler().fit_transform(
            df["text"].str.split().str.len().to_numpy().reshape(-1, 1))
        pred = KMeans(K, n_init=10, random_state=RS).fit_predict(L)

    elif name == "supervised":
        info = "the labels themselves"
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        F, pub = fw.load_features(df, False, [])
        X = np.hstack([S, F[pub].to_numpy(float)])
        cv = StratifiedKFold(5, shuffle=True, random_state=RS)
        acc = 100 * cross_val_score(
            HistGradientBoostingClassifier(random_state=RS), X, y,
            cv=cv).mean()
        return {"method": name, "information": info, "k": K,
                "Match %": round(acc, 2), "ARI": np.nan, "NMI": np.nan,
                "Purity": np.nan, "seconds": round(time.time() - t0)}
    else:
        raise SystemExit(f"unknown method {name}")

    rec = fw.score(y, np.asarray(pred), classes)
    rec.update(method=name, information=info, k=K,
               seconds=round(time.time() - t0))
    gc.collect()
    return rec


PRETTY = {
    "tfidf_kmeans": "TF-IDF + K-Means",
    "lda": "Latent Dirichlet allocation",
    "sbert_kmeans": "Sentence-BERT + K-Means",
    "sbert_umap_kmeans": "Sentence-BERT + UMAP + K-Means",
    "bertopic": "BERTopic (K-Means variant)",
    "proposed": "Proposed two-stream framework",
    "artefact": "Artefact baseline",
    "length": "Document length only",
    "supervised": "Supervised ceiling (5-fold CV)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-slow", action="store_true",
                    help="omit the methods that fit a manifold")
    a = ap.parse_args()

    if a.list:
        for m in METHODS:
            print(f"  {m:<20} {PRETTY[m]}"
                  + ("   [fits a manifold, slower]" if m in SLOW else ""))
        return

    df, S = load()
    print(f"{len(df):,} documents, K = {K}\n")

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["method"])
        print(f"already done: {sorted(done)}\n")

    todo = [a.only] if a.only else [m for m in METHODS
                                    if not (a.skip_slow and m in SLOW)]
    for m in todo:
        if m in done:
            print(f"  {m}: skipped, already recorded")
            continue
        print(f"  {m} ...", flush=True)
        rec = run_method(m, df, S, a)
        if rec is None:
            continue
        _row = pd.DataFrame([rec])
        if os.path.exists(OUT):   # keep the columns aligned with the existing header
            _row = _row.reindex(columns=pd.read_csv(OUT, nrows=0).columns)
        _row.to_csv(OUT, mode="a", index=False, header=not os.path.exists(OUT))
        print(f"    ARI {rec['ARI'] if rec['ARI'] == rec['ARI'] else float('nan'):.4f}"
              f"   purity {rec['Purity'] if rec['Purity'] == rec['Purity'] else float('nan'):.4f}"
              f"   match {rec['Match %']:.2f}%   [{rec['seconds']}s]"
              if rec["ARI"] == rec["ARI"] else
              f"    accuracy {rec['Match %']:.2f}%   [{rec['seconds']}s]",
              flush=True)

    if not os.path.exists(OUT):
        return
    T = pd.read_csv(OUT)
    T["order"] = T["method"].map({m: i for i, m in enumerate(METHODS)})
    T = T.sort_values("order")
    print("\n" + "=" * 78)
    print("  COMPARISON WITH ESTABLISHED UNSUPERVISED METHODS")
    print("=" * 78)
    print(f"{'method':<32}{'information used':<32}{'ARI':>8}{'purity':>9}")
    for _, r in T.iterrows():
        ari = "     —" if pd.isna(pd.to_numeric(r.ARI, errors="coerce")) else f"{float(r.ARI):8.4f}"
        pur = "     —" if pd.isna(pd.to_numeric(r.Purity, errors="coerce")) else f"{float(r.Purity):9.4f}"
        print(f"{PRETTY.get(r.method, r.method):<32}{str(r.information):<32}"
              f"{ari}{pur}")
    print("-" * 78)
    if {"proposed", "sbert_umap_kmeans"} <= set(T.method):
        p = float(T[T.method == "proposed"].ARI.iloc[0])
        s = float(T[T.method == "sbert_umap_kmeans"].ARI.iloc[0])
        print(f"  proposed over the best embedding-only method: {p/s:.2f}x")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
