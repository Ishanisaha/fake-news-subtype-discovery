"""
One-way ANOVA, Kruskal-Wallis and effect sizes (eta squared, omega squared,
length-controlled eta squared, density) for the rhetorical features, the
library features and the artefact features.

    python 02_feature_anova.py
    python 02_feature_anova.py --blocks typography,vader,textblob
    python 02_feature_anova.py --gpu

Output: results/anova_all_features.csv
"""
import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

CORPUS = os.path.join(DATA, "corpus.csv")

SEED = 42
CLASSES = ["propaganda", "rumour", "satire", "clickbait"]

sys.path.insert(0, HERE)


#  the three feature families
def published_features(df):
    """The 13 rhetorical features."""
    from rhetorical_features import extract_rhetorical_features
    d = extract_rhetorical_features(df[["text"]].copy())
    return d[[c for c in d.columns if c.startswith("f_")]]


def artefact_features(texts):
    """Eight statistics with no linguistic content (artefact baseline)."""
    out = []
    for t in texts.astype(str):
        n = max(len(t), 1)
        w = t.split() or [""]
        out.append([t.count(" ") / n, sum(c.isdigit() for c in t) / n,
                    sum(not c.isascii() for c in t) / n,
                    float(np.mean([len(x) for x in w])), t.count("'") / n,
                    float(len([s for s in re.split(r"[.!?]+", t) if s.strip()])),
                    t.count(",") / n, (t.count("#") + t.count("@")) / n])
    cols = ["a_space", "a_digit", "a_nonascii", "a_wordlen", "a_apostrophe",
            "a_nsent", "a_comma", "a_platform"]
    return pd.DataFrame(np.asarray(out, float), columns=cols)


def library_features(texts, device, blocks):
    from library_features import extract_features_v3
    return extract_features_v3(list(texts), device=device, blocks=blocks)


#  the test
def delength(x, nwords, bins=10):
    """Strip out whatever a feature knows purely because of document length."""
    x = np.asarray(x, float)
    q = pd.qcut(pd.Series(nwords).rank(method="first"), bins, labels=False)
    return x - pd.Series(x).groupby(q).transform("mean").to_numpy()


def anova_row(x, y, groups):
    """One feature against the sub-type labels."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, y = x[ok], np.asarray(y)[ok]
    parts = [x[y == g] for g in groups]
    parts = [p for p in parts if len(p) > 1]
    if len(parts) < 2 or np.allclose(x, x[0]):
        return None

    F, p = stats.f_oneway(*parts)
    H, ph = stats.kruskal(*parts)

    grand = x.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in parts)
    ss_t = ((x - grand) ** 2).sum()
    k, N = len(parts), len(x)
    eta = ss_b / ss_t if ss_t else 0.0
    ss_w = ss_t - ss_b
    ms_w = ss_w / max(N - k, 1)
    omega = (ss_b - (k - 1) * ms_w) / (ss_t + ms_w) if (ss_t + ms_w) else 0.0

    means = {g: float(x[y == g].mean()) if (y == g).any() else np.nan
             for g in groups}
    hi = max(means, key=lambda g: means[g])
    lo = min(means, key=lambda g: means[g])
    sd = x.std() or 1.0

    return dict(F=float(F), p=float(p), eta2=float(eta),
                omega2=float(max(omega, 0.0)), H=float(H), p_kw=float(ph),
                density=float((x != 0).mean()),
                separates=hi, against=lo,
                gap_d=float((means[hi] - means[lo]) / sd),
                **{f"mean_{g}": means[g] for g in groups})


def run_family(F, y, family, groups, nwords=None):
    rows = []
    for c in F.columns:
        r = anova_row(F[c].to_numpy(), y, groups)
        if r is None:
            print(f"  {c}: constant - skipped")
            continue
        if nwords is not None:
            rr = anova_row(delength(F[c].to_numpy(), nwords), y, groups)
            r["eta2_nolen"] = rr["eta2"] if rr else 0.0
        rows.append(dict(feature=c, family=family, **r))
    return rows


def band(e):
    return ("large" if e >= 0.14 else "medium" if e >= 0.06 else
            "small" if e >= 0.01 else "negligible")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="sample this many documents (0 = all)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--blocks", default="",
                    help="comma-separated subset of the library blocks")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached feature matrices and re-extract")
    a = ap.parse_args()

    if not os.path.exists(CORPUS):
        raise SystemExit(f"missing {CORPUS} - run 01_clean_corpus.py")
    df = pd.read_csv(CORPUS)
    df["label"] = df["label"].replace({"rumor": "rumour"})
    df["text"] = df["text"].fillna("").astype(str)
    if a.n and a.n < len(df):
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), a.n // 4), random_state=SEED))
    df = df.reset_index(drop=True)
    y = df["label"].to_numpy()
    groups = [c for c in CLASSES if (y == c).any()]

    print("=" * 78)
    print(f"CORPUS: {len(df):,} documents, {len(groups)} sub-types, "
          f"median {df['text'].str.split().str.len().median():.0f} words")
    print(f"        {os.path.basename(CORPUS)}")
    print("=" * 78)

    rows = []
    nw = df["text"].str.split().str.len().to_numpy()
    blocks = [b.strip() for b in a.blocks.split(",") if b.strip()] or None

    def cached(tag, fn):
        """Extraction is cached per family so a long run can be done in
        pieces, and so the transformer blocks are never paid for twice."""
        path = os.path.join(RES, f"_cache_{tag}_{len(df)}.csv.gz")
        if os.path.exists(path) and not a.refresh:
            print(f"      cached: {os.path.basename(path)}")
            return pd.read_csv(path)
        F = fn()
        F.to_csv(path, index=False)
        return F

    print("\n[1/3] published rhetorical features")
    rows += run_family(cached("published", lambda: published_features(df)),
                       y, "published", groups, nw)
    print(f"      {sum(r['family']=='published' for r in rows)} features")

    print("\n[2/3] artefact baseline")
    rows += run_family(cached("artefact",
                              lambda: artefact_features(df["text"])),
                       y, "artefact", groups, nw)

    print("\n[3/3] library features")
    try:
        tag = "library" + ("_" + "-".join(blocks) if blocks else "")
        L = cached(tag, lambda: library_features(df["text"],
                                                 0 if a.gpu else -1, blocks))
        rows += run_family(L, y, "library", groups, nw)
    except SystemExit as e:
        print(f"      none available: {e}")

    R = pd.DataFrame(rows).sort_values("eta2", ascending=False)
    R["effect"] = R["eta2"].map(band)
    out_csv = os.path.join(RES, "anova_all_features.csv")
    R.to_csv(out_csv, index=False)

    # the highest artefact score is the line every real feature has to clear
    floor = float(R[R.family == "artefact"]["eta2"].max())

    print("\n" + "=" * 78)
    print("ANOVA ACROSS THE FOUR SUB-TYPES, RANKED BY EFFECT SIZE")
    print("=" * 78)
    print(f"{'feature':<24}{'family':<11}{'eta2':>8}{'no-len':>8}{'omega2':>9}"
          f"{'F':>9}{'dens':>7}  {'effect':<11}separates")
    print("-" * 86)
    for _, r in R.iterrows():
        mark = " " if r.family == "artefact" else ("*" if r.eta2 > floor
                                                   else " ")
        nl = r.get("eta2_nolen", np.nan)
        print(f"{mark}{r.feature:<23}{r.family:<11}{r.eta2:>8.4f}"
              f"{nl:>8.4f}{r.omega2:>9.4f}{r.F:>9.1f}{100*r.density:>6.0f}%  "
              f"{r.effect:<11}{r.separates}")

    print("-" * 86)
    print(f"  artefact floor: eta2 = {floor:.4f}   "
          f"(* marks features that clear it)")
    print("  no-len: eta2 recomputed within length deciles - what the feature")
    print("          knows that document length does not already tell you")

    print("\n" + "=" * 78)
    print("BY FAMILY")
    print("=" * 78)
    g = R.groupby("family").agg(
        n=("eta2", "size"), mean_eta2=("eta2", "mean"),
        max_eta2=("eta2", "max"),
        mean_nolen=("eta2_nolen", "mean"), max_nolen=("eta2_nolen", "max"),
        above_floor=("eta2", lambda s: int((s > floor).sum())),
        mean_density=("density", "mean")).round(4)
    print(g.to_string())

    print("\n" + "=" * 78)
    print("WHICH SUB-TYPE EACH FAMILY CAN SEE")
    print("=" * 78)
    print("  the feature with the largest effect for each sub-type, by family")
    for fam in ["published", "library"]:
        sub = R[R.family == fam]
        if not len(sub):
            continue
        print(f"\n  {fam}")
        for cls in groups:
            s = sub[sub.separates == cls]
            if len(s):
                b = s.iloc[0]
                print(f"    {cls:<12} {b.feature:<22} eta2 {b.eta2:.4f}  "
                      f"d {b.gap_d:+.2f} vs {b.against}")
            else:
                print(f"    {cls:<12} -- no feature peaks on this sub-type")

    print(f"\nwritten: {out_csv}")
    print("\n  eta2 bands: 0.01 small, 0.06 medium, 0.14 large (Cohen).")
    print("  A feature below the artefact floor is not measuring rhetoric,")
    print("  whatever it is named.")


if __name__ == "__main__":
    main()
