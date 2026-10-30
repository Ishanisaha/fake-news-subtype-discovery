"""
Figures: cluster scatter plots (a separate 2-D UMAP of the fused representation,
coloured by the clusters found in 40 dimensions), confusion matrix and result charts.

    python 12_figures.py --figs clusters
    python 12_figures.py --figs charts

Output: figures/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)
sys.path.insert(0, HERE)


import framework as fw

RS = 42
WS, WR = 1.5, 1.0
K = 4
ALG = "K-Means"

CLASSES = ["propaganda", "rumour", "satire", "clickbait"]

# Journals print the caption below the figure, so a title inside the image
# duplicates it. Set this True for slides or for reading the figures alone.
IN_IMAGE_TITLES = False
COL = {"propaganda": "#c0392b", "rumour": "#2874a6",
       "satire": "#1e8449", "clickbait": "#b7950b"}
INK, GREY = "#1a1a1a", "#777777"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.linewidth": 0.8,
        "figure.facecolor": "white"})
    return plt


def _t(ax, text, loc="left", fontsize=9.5):
    """Title the axes only when the figure is meant to stand alone."""
    if IN_IMAGE_TITLES:
        ax.set_title(text, loc=loc, fontsize=fontsize)


def _st(fig, text, **kw):
    """Figure-level title, likewise only when the figure stands alone."""
    if IN_IMAGE_TITLES:
        fig.suptitle(text, **kw)


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        p = os.path.join(FIGS, f"{name}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    print(f"  {name}.svg / .pdf / .png")


#  the representation, computed once and cached
def prepare():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler, normalize

    df = pd.read_csv(fw.CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    n = len(df)
    cache = os.path.join(DATA, f"sbert_cased_{n}.npy")
    if not os.path.exists(cache):
        raise SystemExit(f"missing {cache} - run framework.py first")
    S = normalize(np.load(cache).astype(np.float32))
    F, pub = fw.load_features(df, False, [])
    R = fw.rff(MinMaxScaler().fit_transform(F[pub].to_numpy(float))
               ).astype(np.float32)
    X = normalize(np.hstack([S * WS, R * WR]))

    tag = f"{n}_{WS:g}-{WR:g}"
    p40 = os.path.join(DATA, f"umap40_{tag}.npy")
    p2 = os.path.join(DATA, f"umap2_{tag}.npy")
    p2s = os.path.join(DATA, f"umap2_semantic_{n}.npy")
    p2r = os.path.join(DATA, f"umap2_rhetorical_{n}.npy")

    def emb(path, M, dims):
        if os.path.exists(path):
            print(f"  cached {os.path.basename(path)}")
            return np.load(path)
        print(f"  computing {os.path.basename(path)} ...", flush=True)
        E = fw.umap40(M, dims=dims)
        np.save(path, E)
        return E

    U40 = emb(p40, X, 40)
    U2 = emb(p2, X, 2)
    U2s = emb(p2s, S, 2)
    U2r = emb(p2r, R, 2)

    y = df["label"].to_numpy()
    classes = [c for c in CLASSES if (y == c).any()]
    pred = (KMeans(K, n_init=10, random_state=RS).fit_predict(U40)
            if ALG == "K-Means" else fw.gmm(U40, K, RS))
    named = fw.align(y, pred, classes)
    return df, y, classes, U2, U2s, U2r, pred, named


#  Fig: the clusters
def fig_clusters(y, classes, U2, pred, named):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    rng = np.random.RandomState(RS)
    order = rng.permutation(len(y))      # so no class is drawn on top

    for ax, colours, title in (
            (axes[0], np.array([COL[c] for c in y]),
             "(a)  true sub-types"),
            (axes[1], np.array([COL[c] for c in named]),
             f"(b)  clusters discovered by the framework")):
        ax.scatter(U2[order, 0], U2[order, 1], c=colours[order], s=0.7,
                   alpha=0.45, linewidths=0, rasterized=True)
        ax.set_title(title, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")

    agree = float((np.asarray(named) == np.asarray(y)).mean())
    axes[1].text(0.98, 0.02, f"{100*agree:.1f}% agreement",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=7.5, color=GREY)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=4.5, color=COL[c],
                          label=c) for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=len(classes),
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    _st(fig, "Fig. 2.  The corpus in two dimensions: true sub-types and "
                 "discovered clusters", x=0.02, ha="left", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save(fig, "Fig_Clusters")
    plt.close(fig)


#  Fig: what each stream sees
def fig_streams(y, classes, U2, U2s, U2r):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    rng = np.random.RandomState(RS)
    order = rng.permutation(len(y))
    colours = np.array([COL[c] for c in y])
    for ax, U, t in ((axes[0], U2s, "(a)  semantic stream"),
                     (axes[1], U2r, "(b)  rhetorical stream"),
                     (axes[2], U2, f"(c)  fused, {WS:g}:{WR:g}")):
        ax.scatter(U[order, 0], U[order, 1], c=colours[order], s=0.6,
                   alpha=0.4, linewidths=0, rasterized=True)
        ax.set_title(t, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4.5, color=COL[c],
                          label=c) for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=len(classes),
               frameon=False, bbox_to_anchor=(0.5, -0.07))
    _st(fig, "Fig. 3.  The same documents under each representation, "
                 "coloured by true sub-type", x=0.02, ha="left", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save(fig, "Fig_Streams")
    plt.close(fig)


#  Fig: confusion and composition
def fig_confusion(y, classes, named):
    plt = style()
    cm = pd.crosstab(pd.Series(named, name="cluster"),
                     pd.Series(y, name="true")).reindex(
        index=classes, columns=classes, fill_value=0)
    row = cm.div(cm.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    im = ax.imshow(row.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels([f"cluster → {c}" for c in classes])
    ax.set_xlabel("true sub-type")
    ax.set_title("(a)  confusion, row-normalised", loc="left")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = row.values[i, j]
            ax.text(j, i, f"{cm.values[i, j]:,}\n{100*v:.0f}%",
                    ha="center", va="center", fontsize=6.6,
                    color="white" if v > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).outline.set_visible(
        False)

    ax = axes[1]
    bottom = np.zeros(len(classes))
    for c in classes:
        vals = (cm[c] / cm.sum(axis=1)).values
        ax.bar(range(len(classes)), vals, bottom=bottom, color=COL[c],
               width=0.62, label=c, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"cluster\n→ {c}" for c in classes])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the cluster")
    ax.set_title("(b)  composition of each cluster", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    prec = np.diag(cm.values) / cm.values.sum(axis=1)
    for i, p in enumerate(prec):
        ax.text(i, 1.02, f"{100*p:.0f}%", ha="center", fontsize=7,
                color=GREY)
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.42))

    _st(fig, f"Fig. 4.  Cluster composition against the held-back "
                 f"sub-types ({ALG}, K = {K})", x=0.02, ha="left",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    save(fig, "Fig_Confusion")
    plt.close(fig)
    return cm


#  Fig: the ablation
def fig_ablation():
    path = os.path.join(RES, f"ablation_k{K}.csv")
    if not os.path.exists(path):
        print("  ablation_k4.csv not found - skipping Fig_Ablation")
        return
    plt = style()
    T = pd.read_csv(path).set_index("condition")
    order = ["fused", "semantic", "rhetorical", "rhetorical_raw",
             "fused_norbf"]
    order = [c for c in order if c in T.index]
    label = {"fused": f"Fused  {WS:g}:{WR:g}\n(proposed)",
             "semantic": "Semantic\nonly",
             "rhetorical": "Rhetorical\nonly (RBF)",
             "rhetorical_raw": "Rhetorical\nonly (raw)",
             "fused_norbf": "Fused\nwithout RBF"}

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    x = np.arange(len(order))
    w = 0.38
    km = [T.loc[c, "ARI_kmeans"] for c in order]
    gm = [T.loc[c, "ARI_gmm"] for c in order]
    b1 = ax.bar(x - w / 2, km, w, label="K-Means", color="#1f4e79",
                edgecolor="none")
    b2 = ax.bar(x + w / 2, gm, w, label="GMM", color="#8fb3d1",
                edgecolor="none")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.3f}", ha="center", fontsize=6.8,
                    color=INK)

    if "artefact" in T.index:
        v = float(T.loc["artefact", "ARI_kmeans"])
        ax.axhline(v, ls="--", lw=1, color="#c0392b")
        ax.text(len(order) - 0.42, v + 0.008,
                f"artefact floor  {v:.3f}", fontsize=7, color="#c0392b",
                ha="right")
    if "length" in T.index:
        v = float(T.loc["length", "ARI_kmeans"])
        ax.axhline(v, ls=":", lw=1.2, color="#7d6608")
        ax.text(len(order) - 0.42, v + 0.008,
                f"document length alone  {v:.3f}", fontsize=7,
                color="#7d6608", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([label[c] for c in order])
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(0, max(km + gm) * 1.22)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _t(ax, "Fig. 5.  Ablation over the input representation.")
    fig.tight_layout()
    save(fig, "Fig_Ablation")
    plt.close(fig)


#  Fig: the weight sweep
def fig_weights():
    path = os.path.join(RES, f"weight_sweep_published_k{K}.csv")
    if not os.path.exists(path):
        print("  weight_sweep not found - skipping Fig_Weights")
        return
    plt = style()
    T = pd.read_csv(path).copy()
    # order the configurations along a single axis running from all-semantic
    # to all-rhetorical, which is what the sweep actually traverses
    T["ratio"] = T.w_s / T.w_r.replace(0, np.nan)
    T["rank"] = T.apply(
        lambda r: 1e6 if r.w_r == 0 else (-1e6 if r.w_s == 0 else r.ratio),
        axis=1)
    T = T.sort_values("rank", ascending=False).reset_index(drop=True)
    lab = [("semantic\nonly" if r.w_r == 0 else
            "rhetorical\nonly" if r.w_s == 0 else f"{r.w_s:g}:{r.w_r:g}")
           for _, r in T.iterrows()]
    x = np.arange(len(T))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(x, T.ARI_kmeans, "-o", ms=4, lw=1.4, color="#1f4e79",
            label="K-Means")
    ax.plot(x, T.ARI_gmm, "-s", ms=3.6, lw=1.2, color="#8fb3d1",
            label="GMM")
    if "ARI_kmeans_sd" in T.columns:
        ax.fill_between(x, T.ARI_kmeans - T.ARI_kmeans_sd.fillna(0),
                        T.ARI_kmeans + T.ARI_kmeans_sd.fillna(0),
                        color="#1f4e79", alpha=0.12, linewidth=0)

    best = T.ARI_kmeans.idxmax()
    ax.annotate(f"reported configuration\n{T.loc[best,'w_s']:g}:"
                f"{T.loc[best,'w_r']:g}   ARI {T.loc[best,'ARI_kmeans']:.4f}",
                xy=(best, T.loc[best, "ARI_kmeans"]),
                xytext=(best + 0.6, T.loc[best, "ARI_kmeans"] + 0.03),
                fontsize=7.2, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=GREY))

    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_xlabel("semantic weight : rhetorical weight")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(0, max(T.ARI_kmeans.max(), T.ARI_gmm.max()) * 1.2)
    ax.legend(frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _t(ax, "Fig. 6.  Sensitivity to the stream weighting.")
    fig.tight_layout()
    save(fig, "Fig_Weights")
    plt.close(fig)


#  Fig: comparison with established methods
def fig_sota():
    path = os.path.join(RES, "sota_comparison.csv")
    if not os.path.exists(path):
        print("  sota_comparison.csv not found - skipping Fig_SOTA")
        return
    plt = style()
    T = pd.read_csv(path).set_index("method")
    order = ["lda", "tfidf_kmeans", "sbert_kmeans", "sbert_umap_kmeans",
             "bertopic", "proposed"]
    order = [m for m in order if m in T.index and pd.notna(T.loc[m, "ARI"])]
    label = {"lda": "LDA", "tfidf_kmeans": "TF-IDF\n+ K-Means",
             "sbert_kmeans": "SBERT\n+ K-Means",
             "sbert_umap_kmeans": "SBERT + UMAP\n+ K-Means",
             "bertopic": "BERTopic", "proposed": "Proposed\ntwo-stream"}

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(order))
    w = 0.38
    ari = [float(T.loc[m, "ARI"]) for m in order]
    pur = [float(T.loc[m, "Purity"]) for m in order]
    cols = ["#8fb3d1"] * len(order)
    if "proposed" in order:
        cols[order.index("proposed")] = "#1f4e79"
    b1 = ax.bar(x - w / 2, ari, w, color=cols, edgecolor="none",
                label="Adjusted Rand Index")
    b2 = ax.bar(x + w / 2, pur, w, color="#c9d8e6", edgecolor="none",
                label="Purity")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.3f}", ha="center", fontsize=6.6,
                    color=INK)

    for key, ls, col, txt in (("artefact", "--", "#c0392b",
                               "artefact floor"),
                              ("length", ":", "#7d6608",
                               "document length alone")):
        if key in T.index and pd.notna(T.loc[key, "ARI"]):
            v = float(T.loc[key, "ARI"])
            ax.axhline(v, ls=ls, lw=1.0, color=col)
            ax.text(len(order) - 0.45, v + 0.008, f"{txt}  {v:.3f}",
                    fontsize=7, color=col, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([label[m] for m in order], fontsize=7.2)
    ax.set_ylabel("agreement with the held-back sub-types")
    ax.set_ylim(0, max(ari + pur) * 1.20)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    _t(ax, "Fig. 8.  Comparison with established unsupervised methods.")
    fig.tight_layout()
    save(fig, "Fig_SOTA")
    plt.close(fig)


#  Fig: cross-validation
def fig_crossval():
    f_folds = os.path.join(RES, "cv_folds.csv")
    f_coll = os.path.join(RES, "cv_collections.csv")
    f_ext = os.path.join(RES, "cv_external.csv")
    if not os.path.exists(f_folds):
        print("  cv_folds.csv not found - skipping Fig_CrossValidation")
        return
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))

    T = pd.read_csv(f_folds)
    ax = axes[0]
    for algo, col in (("kmeans", "#1f4e79"), ("gmm", "#8fb3d1")):
        s = T[T.algorithm == algo]
        tr = s[s.split == "training"].sort_values("fold")
        te = s[s.split == "held-out"].sort_values("fold")
        ax.plot(tr.fold, tr.ARI, "--o", ms=3.5, lw=1, color=col, alpha=0.55,
                label=f"{algo}, training")
        ax.plot(te.fold, te.ARI, "-o", ms=4, lw=1.4, color=col,
                label=f"{algo}, held-out")
    ax.set_xlabel("fold")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(T.fold.unique()))
    ax.set_title("(a)  five-fold, unseen documents", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")

    ax = axes[1]
    if os.path.exists(f_coll):
        C = pd.read_csv(f_coll)
        C = C[C.algorithm == ("kmeans" if ALG == "K-Means" else "gmm")]
        pairs, vals, cols = [], [], []
        for _, r in C.iterrows():
            same = r.fitted_on == r.applied_to
            pairs.append(f"{r.fitted_on.replace('collection ','C')}\n→"
                         f"{r.applied_to.replace('collection ','C')}")
            vals.append(r.ARI)
            cols.append("#b8c9d9" if same else "#1f4e79")
        ax.bar(range(len(vals)), vals, color=cols, width=0.62)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(pairs, fontsize=6.8)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=6.6)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title("(b)  transfer across collections", loc="left")
        ax.text(0.02, 0.95, "pale = in-sample", transform=ax.transAxes,
                fontsize=6.4, color=GREY, va="top")
    ax.set_ylabel("Adjusted Rand Index")

    ax = axes[2]
    if os.path.exists(f_ext):
        X = pd.read_csv(f_ext)
        X = X[X.algorithm == ("kmeans" if ALG == "K-Means" else "gmm")]
        names = [str(t).replace(" subset", "").replace("labeller ", "")
                 .replace(" (", "\n(") for t in X.target] \
            if "target" in X.columns else ["external"] * len(X)
        vals = list(X.ARI)
        err = list(X.ARI_sd.fillna(0)) if "ARI_sd" in X.columns \
            else [0] * len(X)
        ax.bar(range(len(vals)), vals, yerr=err, capsize=2.5,
               color="#2e7d5b", width=0.6,
               error_kw=dict(lw=0.8, ecolor=GREY))
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, fontsize=6.2, rotation=20, ha="right")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.4)
        ax.set_ylim(0, max(max(vals) * 1.35, 0.35))
        # with a single target the bar would otherwise fill the panel, which
        # reads as a much larger result than it is
        ax.set_xlim(-0.75, len(vals) - 0.25)
        if "kappa_between_labellers" in X.columns:
            ax.text(0.02, 0.95,
                    f"labeller κ = {X.kappa_between_labellers.iloc[0]:.3f}",
                    transform=ax.transAxes, fontsize=6.4, color=GREY,
                    va="top")
    ax.set_title("(c)  external corpora", loc="left")

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    _st(fig, "Fig. 7.  Cross-validation: generalisation to unseen "
                 "documents, to the other collection, and beyond the corpus",
                 x=0.02, ha="left", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "Fig_CrossValidation")
    plt.close(fig)


#  editable PowerPoint of the two bar/line figures
def pptx_charts():
    try:
        from pptx import Presentation
        from pptx.chart.data import CategoryChartData
        from pptx.enum.chart import XL_CHART_TYPE
        from pptx.util import Inches, Pt
    except ImportError:
        print("  python-pptx not installed - skipped the editable deck")
        return
    abl = os.path.join(RES, f"ablation_k{K}.csv")
    swp = os.path.join(RES, f"weight_sweep_published_k{K}.csv")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(10), Inches(5.6)

    if os.path.exists(abl):
        T = pd.read_csv(abl).set_index("condition")
        order = [c for c in ["fused", "semantic", "rhetorical",
                             "rhetorical_raw", "fused_norbf", "artefact",
                             "length"] if c in T.index]
        d = CategoryChartData()
        d.categories = [T.loc[c, "name"] for c in order]
        d.add_series("K-Means", [float(T.loc[c, "ARI_kmeans"])
                                 for c in order])
        d.add_series("GMM", [float(T.loc[c, "ARI_gmm"]) for c in order])
        sl = prs.slides.add_slide(prs.slide_layouts[5])
        sl.shapes.title.text = "Ablation over the input representation"
        sl.shapes.title.text_frame.paragraphs[0].runs[0].font.size = Pt(20)
        sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4),
                            Inches(1.2), Inches(9.2), Inches(4.0), d)

    if os.path.exists(swp):
        T = pd.read_csv(swp)
        T["rank"] = T.apply(lambda r: 1e6 if r.w_r == 0 else
                            (-1e6 if r.w_s == 0 else r.w_s / r.w_r), axis=1)
        T = T.sort_values("rank", ascending=False)
        d = CategoryChartData()
        d.categories = [f"{r.w_s:g}:{r.w_r:g}" for _, r in T.iterrows()]
        d.add_series("K-Means", [float(v) for v in T.ARI_kmeans])
        d.add_series("GMM", [float(v) for v in T.ARI_gmm])
        sl = prs.slides.add_slide(prs.slide_layouts[5])
        sl.shapes.title.text = "Sensitivity to the stream weighting"
        sl.shapes.title.text_frame.paragraphs[0].runs[0].font.size = Pt(20)
        sl.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.4),
                            Inches(1.2), Inches(9.2), Inches(4.0), d)

    out = os.path.join(FIGS, "Fig_Results_editable.pptx")
    prs.save(out)
    print(f"  Fig_Results_editable.pptx  (native charts, data editable)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default="all",
                    choices=["all", "clusters", "charts"])
    a = ap.parse_args()

    if a.figs in ("all", "charts"):
        print("charts from the result files:")
        fig_ablation()
        fig_weights()
        fig_sota()
        fig_crossval()
        pptx_charts()

    if a.figs in ("all", "clusters"):
        print("\ncluster figures (embeddings are cached after the first run):")
        df, y, classes, U2, U2s, U2r, pred, named = prepare()
        fig_clusters(y, classes, U2, pred, named)
        fig_streams(y, classes, U2, U2s, U2r)
        cm = fig_confusion(y, classes, named)
        cm.to_csv(os.path.join(RES, f"confusion_{WS:g}-{WR:g}_"
                                    f"{ALG.lower()}.csv"))
        print("\n  confusion matrix:")
        print(cm.to_string())

    print(f"\nfigures written to {FIGS}")


if __name__ == "__main__":
    main()
