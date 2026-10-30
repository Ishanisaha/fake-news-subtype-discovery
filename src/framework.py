"""
Shared pipeline components: corpus paths, rhetorical features, Sentence-BERT
embeddings, random Fourier features, SVD + UMAP reduction, clustering and scoring.

Imported by the numbered scripts. It can also be run on its own for a single
end-to-end run of the framework:

    python framework.py                 # fused 1.5 : 1, K-Means, K = 4
    python framework.py --sweep         # stream-weight sweep
    python framework.py --ksweep        # K = 2..10
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                             homogeneity_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, HERE)

CORPUS = os.path.join(DATA, "corpus.csv")

RS = 42
SVD_DIMS, UMAP_NEIGH, UMAP_EPOCHS, UMAP_DIMS = 100, 15, 200, 40
RBF_DIMS, RBF_GAMMA = 200, 1.0
WS, WR = 1.5, 1.0
SBERT_MODEL = "all-MiniLM-L6-v2"
CLASSES = ["propaganda", "rumour", "satire", "clickbait"]

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


#  streams
def rff(x, seed=RS, D=RBF_DIMS, g=RBF_GAMMA):
    """Random Fourier Features: an explicit, finite approximation to an RBF
    kernel map. Used instead of a kernel matrix because 17,465 squared is a
    matrix nobody needs to build."""
    rng = np.random.RandomState(seed)
    return normalize(np.sqrt(2 / D) * np.cos(
        x @ rng.normal(0, np.sqrt(2 * g), (x.shape[1], D))
        + rng.uniform(0, 2 * np.pi, D)))


def umap40(E, dims=UMAP_DIMS):
    import umap.umap_ as umap
    Z = TruncatedSVD(min(SVD_DIMS, E.shape[1] - 1),
                     random_state=RS).fit_transform(E)
    return umap.UMAP(n_neighbors=UMAP_NEIGH,
                     n_components=min(dims, Z.shape[1] - 1), min_dist=0.0,
                     metric="cosine", random_state=RS, low_memory=True,
                     n_epochs=UMAP_EPOCHS, n_jobs=1).fit_transform(Z)


def sbert(texts, cache_tag, fake=False):
    """Encode once, cache, reuse. The cache key includes the document count so
    a corpus change cannot silently reuse the wrong embeddings."""
    path = os.path.join(DATA, f"sbert_{cache_tag}.npy")
    if os.path.exists(path):
        E = np.load(path)
        say(f"  SBERT: cached {E.shape}")
        return E
    if fake:
        say("  SBERT: FAKE random embeddings - plumbing test only, "
            "the numbers mean nothing")
        return np.random.RandomState(RS).normal(size=(len(texts), 384))
    from sentence_transformers import SentenceTransformer
    say(f"  SBERT: encoding {len(texts):,} documents with {SBERT_MODEL}")
    t0 = time.time()
    E = SentenceTransformer(SBERT_MODEL).encode(
        list(texts), show_progress_bar=True, batch_size=64)
    say(f"  SBERT: {time.time()-t0:.0f}s")
    np.save(path, E)
    return E


#  scoring
def purity(y, p):
    cm = pd.crosstab(p, y).values
    return cm.max(axis=1).sum() / cm.sum()


def align(y, pred, classes):
    """Hungarian matching."""
    cm = confusion_matrix(pd.Categorical(y, classes).codes, pred,
                          labels=range(max(len(classes), int(pred.max()) + 1)))
    r, c = linear_sum_assignment(-cm)
    m = {int(cc): classes[int(rr)] for rr, cc in zip(r, c) if rr < len(classes)}
    return np.array([m.get(int(x), classes[0]) for x in pred])


def score(y, pred, classes, **extra):
    al = align(y, pred, classes)
    d = {"Match %": round(100 * float((al == y).mean()), 2),
         "ARI": round(adjusted_rand_score(y, pred), 4),
         "NMI": round(normalized_mutual_info_score(y, pred), 4),
         "Homogeneity": round(homogeneity_score(y, pred), 4),
         "Purity": round(purity(y, pred), 4)}
    d.update(extra)
    return d


def gmm(U, k, seed):
    return GaussianMixture(k, covariance_type="diag", n_init=5,
                           reg_covar=1e-4, random_state=seed).fit_predict(U)


#  feature selection
def pick_features(F, mode, published_cols):
    """Which columns go into the rhetorical stream."""
    if mode == "published":
        return published_cols
    if mode == "all":
        return list(F.columns)
    if mode.startswith("top"):
        n = int(mode[3:])
        path = os.path.join(RES, "anova_all_features.csv")
        if not os.path.exists(path):
            raise SystemExit("run 02_feature_anova.py first - "
                             f"{mode} ranks on its output")
        R = pd.read_csv(path)
        R = R[R.family != "artefact"]
        col = "eta2_nolen" if "eta2_nolen" in R.columns else "eta2"
        keep = [c for c in R.sort_values(col, ascending=False)["feature"]
                if c in F.columns][:n]
        say(f"  top {n} by length-controlled effect size: {', '.join(keep)}")
        return keep
    raise SystemExit(f"unknown --features {mode}")


def load_features(df, want_library, blocks, refresh=False):
    """Load cached feature matrices if present, otherwise extract and cache them."""
    from rhetorical_features import extract_rhetorical_features
    n = len(df)
    parts, published_cols = [], []

    p = os.path.join(RES, f"_cache_published_{n}.csv.gz")
    if os.path.exists(p) and not refresh:
        Fp = pd.read_csv(p)
        say(f"  features: cached published ({Fp.shape[1]})")
    else:
        d = extract_rhetorical_features(df[["text"]].copy())
        Fp = d[[c for c in d.columns if c.startswith("f_")]]
        Fp.to_csv(p, index=False)
        say(f"  features: extracted published ({Fp.shape[1]})")
    published_cols = list(Fp.columns)
    parts.append(Fp)

    if want_library:
        tag = "library" + ("_" + "-".join(blocks) if blocks else "")
        p = os.path.join(RES, f"_cache_{tag}_{n}.csv.gz")
        if os.path.exists(p) and not refresh:
            Fl = pd.read_csv(p)
            say(f"  features: cached library ({Fl.shape[1]})")
        else:
            from library_features import extract_features_v3
            Fl = extract_features_v3(list(df["text"]), device=-1,
                                     blocks=blocks or None)
            Fl.to_csv(p, index=False)
        # typography duplicates two published features by construction
        Fl = Fl[[c for c in Fl.columns if c not in published_cols]]
        parts.append(Fl)

    F = pd.concat(parts, axis=1)
    return F.loc[:, ~F.columns.duplicated()], published_cols


def artefact_features(texts):
    import re
    out = []
    for t in texts.astype(str):
        n = max(len(t), 1)
        w = t.split() or [""]
        out.append([t.count(" ") / n, sum(c.isdigit() for c in t) / n,
                    sum(not c.isascii() for c in t) / n,
                    float(np.mean([len(x) for x in w])), t.count("'") / n,
                    float(len([s for s in re.split(r"[.!?]+", t) if s.strip()])),
                    t.count(",") / n, (t.count("#") + t.count("@")) / n])
    return np.asarray(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="published",
                    help="published | all | top8 (any topN)")
    ap.add_argument("--blocks", default="",
                    help="library blocks to use when --features all")
    ap.add_argument("--ws", type=float, default=WS)
    ap.add_argument("--wr", type=float, default=WR)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sample", type=int, default=0,
                    help="documents per sub-type (0 = all)")
    ap.add_argument("--sweep", action="store_true", help="weight sensitivity")
    ap.add_argument("--ksweep", action="store_true", help="k = 2..10 with BIC")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--algo", default="kmeans", choices=["gmm", "kmeans"],
                    help="which algorithm the confusion matrix comes from")
    ap.add_argument("--fake-sbert", action="store_true",
                    help="random vectors instead of SBERT. Checks that the "
                         "pipeline runs; the numbers are meaningless.")
    a = ap.parse_args()

    from rhetorical_features import clean_text_aggressive

    df = pd.read_csv(CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    if a.sample:
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(a.sample, len(g)),
                                          random_state=RS))
                .sample(frac=1, random_state=RS))
    df = df.reset_index(drop=True)

    say("=" * 78)
    say("  TWO-STREAM RHETORICAL-SEMANTIC FRAMEWORK")
    say("=" * 78)
    say(f"  corpus     {os.path.basename(CORPUS)}  {len(df):,} documents")
    say(f"  weights    semantic {a.ws:g} : rhetorical {a.wr:g}")
    say(f"  features   {a.features}")
    say("")

    y = df["label"].to_numpy()
    classes = [c for c in CLASSES if (y == c).any()]
    K = a.k

    blocks = [b.strip() for b in a.blocks.split(",") if b.strip()]
    F, published_cols = load_features(df, a.features != "published", blocks,
                                      a.refresh)
    cols = pick_features(F, a.features, published_cols)
    say(f"  rhetorical stream: {len(cols)} features")

    clean = df["text"].map(clean_text_aggressive)
    E = sbert(clean, f"cased_{len(df)}", fake=a.fake_sbert)

    S = normalize(np.asarray(E, dtype=np.float32))
    Rraw = MinMaxScaler().fit_transform(F[cols].to_numpy(float))
    R = rff(Rraw)
    FUSED = normalize(np.hstack([S * a.ws, R * a.wr]))
    ART = normalize(MinMaxScaler().fit_transform(artefact_features(df["text"])))
    LEN = MinMaxScaler().fit_transform(
        df["text"].str.split().str.len().to_numpy().reshape(-1, 1))

    rows = []

    say("\n" + "=" * 78)
    say("  REPRESENTATIONS - same pipeline, same documents, same seeds")
    say("=" * 78)
    reps = [("Fused (proposed)", FUSED),
            ("Semantic only", S),
            ("Rhetorical only", R),
            ("Artefact baseline", ART),
            ("Length only", LEN)]
    for name, X in reps:
        t0 = time.time()
        U = umap40(X) if X.shape[1] > 2 else X
        for s in range(a.seeds):
            rows.append(score(y, gmm(U, K, RS + s), classes,
                              Representation=name, Algorithm="GMM",
                              k=K, seed=RS + s))
            rows.append(score(y, KMeans(K, n_init=10, random_state=RS + s)
                              .fit_predict(U), classes,
                              Representation=name, Algorithm="K-Means",
                              k=K, seed=RS + s))
        m = pd.DataFrame([r for r in rows if r["Representation"] == name])
        for alg, g in m.groupby("Algorithm"):
            say(f"  {name:<20}{alg:<9} ARI {g.ARI.mean():.4f} "
                f"+/- {g.ARI.std():.4f}   NMI {g.NMI.mean():.4f}   "
                f"purity {g.Purity.mean():.4f}   "
                f"match {g['Match %'].mean():.2f}%")
        say(f"  {'':<20}[{time.time()-t0:.0f}s]")
        del U
        gc.collect()

    say("\n" + "=" * 78)
    say("  CLUSTERING ALGORITHMS ON THE FUSED REPRESENTATION")
    say("=" * 78)
    U = umap40(FUSED)
    alg_rows = []
    for s in range(a.seeds):
        alg_rows.append(score(y, gmm(U, K, RS + s), classes,
                              Algorithm="GMM (proposed)", seed=RS + s))
        alg_rows.append(score(y, KMeans(K, n_init=10, random_state=RS + s)
                              .fit_predict(U), classes,
                              Algorithm="K-Means", seed=RS + s))
    alg_rows.append(score(y, AgglomerativeClustering(K).fit_predict(U),
                          classes, Algorithm="Agglomerative", seed=RS))
    A = pd.DataFrame(alg_rows)
    say(A.groupby("Algorithm")[["ARI", "NMI", "Purity", "Match %"]]
        .mean().round(4).to_string())

    say("\n" + "=" * 78)
    say("  SUPERVISED CEILING - what these features hold when labels are seen")
    say("=" * 78)
    for name, X in [("rhetorical features", F[cols].to_numpy(float)),
                    ("length alone", LEN)]:
        acc = 100 * cross_val_score(
            HistGradientBoostingClassifier(random_state=RS), X, y,
            cv=5, scoring="accuracy").mean()
        say(f"  {name:<22} {acc:6.2f}%   (chance {100/len(classes):.2f}%)")
        rows.append({"Representation": f"supervised: {name}",
                     "Algorithm": "HistGB (5-fold CV)", "Match %": round(acc, 2),
                     "ARI": np.nan, "NMI": np.nan, "Purity": np.nan,
                     "Homogeneity": np.nan, "k": np.nan, "seed": RS})

    best = (gmm(U, K, RS) if a.algo == "gmm"
            else KMeans(K, n_init=10, random_state=RS).fit_predict(U))
    al = align(y, best, classes)
    cm = pd.crosstab(pd.Series(al, name="cluster (matched)"),
                     pd.Series(y, name="true sub-type"))
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)
    say("\n" + "=" * 78)
    say(f"  CONFUSION MATRIX - fused, {a.algo.upper()}, seed 42")
    say("=" * 78)
    say(cm.to_string())
    prec = (np.diag(cm.values) / cm.values.sum(axis=1)).round(3)
    say("\n  per-cluster precision: " +
        "  ".join(f"{c} {p:.3f}" for c, p in zip(classes, prec)))
    cm.to_csv(os.path.join(RES, f"framework_confusion_{a.features}.csv"))

    if a.sweep:
        say("\n" + "=" * 78)
        say("  WEIGHT SENSITIVITY - semantic : rhetorical")
        say("=" * 78)
        sw = []
        for ws, wr in [(1, 0), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (0, 1)]:
            X = normalize(np.hstack([S * ws, R * wr]))
            Uw = umap40(X)
            m = [score(y, gmm(Uw, K, RS + s), classes) for s in range(2)]
            m = pd.DataFrame(m).mean(numeric_only=True)
            sw.append({"w_s": ws, "w_r": wr, "ARI": round(m.ARI, 4),
                       "NMI": round(m.NMI, 4), "Purity": round(m.Purity, 4),
                       "Match %": round(m["Match %"], 2)})
            say(f"  {ws}:{wr:<4} ARI {m.ARI:.4f}   NMI {m.NMI:.4f}   "
                f"purity {m.Purity:.4f}")
            del X, Uw
            gc.collect()
        pd.DataFrame(sw).to_csv(
            os.path.join(RES, f"framework_weights_{a.features}.csv"),
            index=False)

    if a.ksweep:
        say("\n" + "=" * 78)
        say("  MODEL SELECTION - k chosen without looking at the labels")
        say("=" * 78)
        ks = []
        for k in range(2, 11):
            g = GaussianMixture(k, covariance_type="diag", n_init=5,
                                reg_covar=1e-4, random_state=RS).fit(U)
            p = g.predict(U)
            sil = silhouette_score(U, p, sample_size=min(5000, len(U)),
                                   random_state=RS)
            ks.append({"k": k, "BIC": round(g.bic(U), 1),
                       "silhouette": round(float(sil), 4),
                       "ARI": round(adjusted_rand_score(y, p), 4)})
            say(f"  k={k:<3} BIC {g.bic(U):12.1f}   silhouette {sil:.4f}   "
                f"ARI {adjusted_rand_score(y, p):.4f}")
        pd.DataFrame(ks).to_csv(
            os.path.join(RES, f"framework_ksweep_{a.features}.csv"),
            index=False)

    out = pd.DataFrame(rows)
    tag = f"{a.features}_{a.ws:g}-{a.wr:g}_k{K}"
    out.to_csv(os.path.join(RES, f"framework_{tag}.csv"), index=False)
    A.to_csv(os.path.join(RES, f"framework_algorithms_{tag}.csv"), index=False)

    summ = out[out.ARI.notna()].groupby("Representation")[
        ["ARI", "NMI", "Purity", "Match %"]].agg(["mean", "std"]).round(4)
    say("\n" + "=" * 78)
    say("  SUMMARY")
    say("=" * 78)
    say(summ.to_string())

    fused = out[out.Representation == "Fused (proposed)"].ARI.mean()
    art = out[out.Representation == "Artefact baseline"].ARI.mean()
    ln = out[out.Representation == "Length only"].ARI.mean()
    say("")
    say(f"  fused / artefact floor : {fused/max(art,1e-9):.2f}x")
    say(f"  fused / length only    : {fused/max(ln,1e-9):.2f}x")
    if fused <= max(art, ln):
        say("  THE FRAMEWORK DOES NOT CLEAR ITS OWN BASELINES ON THIS "
            "CONFIGURATION.")

    with open(os.path.join(RES, f"framework_{tag}.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(LOG))
    say(f"\nwritten: results/framework_{tag}.csv, .txt, confusion, algorithms")


if __name__ == "__main__":
    main()
