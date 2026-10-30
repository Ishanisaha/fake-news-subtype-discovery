"""
Library-based features used in the feature analysis (02_feature_anova.py):
transformer emotion and irony classifiers, VADER sentiment, TextBlob polarity and
subjectivity, spaCy syntax counts and textstat readability.

    pip install transformers torch spacy textblob vaderSentiment textstat
    python -m spacy download en_core_web_sm

A block whose library is missing is skipped with a message.
"""
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness",
            "surprise"]


#  block 1 - emotion, from a classifier
def emotion_features(texts, batch=32, device=-1):
    cols = [f"f_emo_{e}" for e in EMOTIONS]
    try:
        from transformers import pipeline
    except ImportError:
        print("  emotion: transformers not installed - skipped")
        return None
    print(f"  emotion: {EMOTION_MODEL}", flush=True)
    pipe = pipeline("text-classification", model=EMOTION_MODEL,
                    top_k=None, device=device, truncation=True, max_length=128)
    out = []
    for i in range(0, len(texts), batch):
        for res in pipe(list(texts[i:i + batch])):
            d = {r["label"].lower(): r["score"] for r in res}
            out.append([d.get(e, 0.0) for e in EMOTIONS])
        if i and i % (batch * 20) == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    return pd.DataFrame(out, columns=cols)


#  block 2 - irony, aimed at satire
def irony_features(texts, batch=32, device=-1):
    try:
        from transformers import pipeline
    except ImportError:
        print("  irony: transformers not installed - skipped")
        return None
    print(f"  irony: {IRONY_MODEL}", flush=True)
    pipe = pipeline("text-classification", model=IRONY_MODEL,
                    top_k=None, device=device, truncation=True, max_length=128)
    out = []
    for i in range(0, len(texts), batch):
        for res in pipe(list(texts[i:i + batch])):
            # the model labels these LABEL_1 / irony and LABEL_0 / non_irony
            sc = {r["label"].lower(): r["score"] for r in res}
            out.append([sc.get("irony", sc.get("label_1", 0.0))])
    return pd.DataFrame(out, columns=["f_irony"])


#  block 3 - sentiment, VADER
def vader_features(texts):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
        except ImportError:
            print("  vader: not installed - skipped")
            return None
    print("  sentiment: VADER", flush=True)
    an = SentimentIntensityAnalyzer()
    out = [[(s := an.polarity_scores(str(t)))["pos"], s["neg"], s["neu"],
            s["compound"]] for t in texts]
    return pd.DataFrame(out, columns=["f_vader_pos", "f_vader_neg",
                                      "f_vader_neu", "f_vader_compound"])


#  block 4 - subjectivity, TextBlob
def textblob_features(texts):
    try:
        from textblob import TextBlob
    except ImportError:
        print("  textblob: not installed - skipped")
        return None
    print("  subjectivity: TextBlob", flush=True)
    out = []
    for t in texts:
        b = TextBlob(str(t)).sentiment
        out.append([b.polarity, b.subjectivity])
    return pd.DataFrame(out, columns=["f_tb_polarity", "f_tb_subjectivity"])


#  block 5 - syntax and named entities, spaCy
def spacy_features(texts, batch=64):
    cols = ["f_superlative", "f_comparative", "f_imperative_v", "f_modal_pos",
            "f_passive", "f_ent_person", "f_ent_org", "f_ent_gpe",
            "f_pron_1st", "f_pron_2nd", "f_pron_3rd"]
    try:
        import spacy
    except ImportError:
        print("  spacy: not installed - skipped")
        return None
    try:
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except OSError:
        print("  spacy: model missing - run "
              "'python -m spacy download en_core_web_sm' - skipped")
        return None
    print("  syntax and entities: spaCy en_core_web_sm", flush=True)

    out = []
    for doc in nlp.pipe([str(t) for t in texts], batch_size=batch):
        n = max(len(doc), 1)
        tags = [t.tag_ for t in doc]
        # a clause whose root is an untensed verb with no subject is
        # imperative in the sense that matters here ("watch this", "look")
        imp = sum(1 for s in doc.sents
                  if len(s) and s[0].pos_ == "VERB"
                  and s[0].morph.get("VerbForm") == ["Inf"])
        pr = {"1": 0, "2": 0, "3": 0}
        for t in doc:
            if t.pos_ == "PRON":
                p = t.morph.get("Person")
                if p and p[0] in pr:
                    pr[p[0]] += 1
        ents = {"PERSON": 0, "ORG": 0, "GPE": 0}
        for e in doc.ents:
            if e.label_ in ents:
                ents[e.label_] += 1
        out.append([
            (tags.count("JJS") + tags.count("RBS")) / n,
            (tags.count("JJR") + tags.count("RBR")) / n,
            imp / max(len(list(doc.sents)), 1),
            tags.count("MD") / n,
            sum(1 for t in doc if t.dep_ in ("auxpass", "nsubjpass")) / n,
            ents["PERSON"] / n, ents["ORG"] / n, ents["GPE"] / n,
            pr["1"] / n, pr["2"] / n, pr["3"] / n,
        ])
    return pd.DataFrame(out, columns=cols)


#  block 6 - readability
def readability_features(texts):
    try:
        import textstat
    except ImportError:
        print("  textstat: not installed - skipped")
        return None
    print("  readability: textstat", flush=True)
    out = []
    for t in texts:
        s = str(t) or " "
        try:
            out.append([textstat.flesch_reading_ease(s),
                        textstat.flesch_kincaid_grade(s),
                        textstat.difficult_words(s) / max(len(s.split()), 1)])
        except Exception:                                  # noqa: BLE001
            out.append([0.0, 0.0, 0.0])
    return pd.DataFrame(out, columns=["f_read_ease", "f_read_grade",
                                      "f_hard_words"])


#  block 7 - typography, no library needed
def typography_features(texts):
    print("  typography: built in", flush=True)
    out = []
    for t in texts:
        s = str(t)
        n = max(len(s), 1)
        w = s.split() or [""]
        out.append([
            len(re.findall(r"[!?]{2,}", s)) / n,          # !!! or ?!
            s.count("!") / n,
            s.count("?") / n,
            sum(1 for x in w if len(x) > 1 and x.isupper()) / len(w),
            len(re.findall(r"\.\.\.|…", s)) / n,
        ])
    return pd.DataFrame(out, columns=["f_rep_punct", "f_excl", "f_quest",
                                      "f_allcaps", "f_ellipsis"])


def _safe(name, fn):
    """Run a block, and turn any failure into a skip rather than a crash."""
    try:
        return fn()
    except Exception as e:                                 # noqa: BLE001
        msg = str(e).strip().split("\n")[0][:110]
        print(f"  {name}: skipped - {type(e).__name__}: {msg}")
        if "nltk" in str(e).lower() or "lexicon" in str(e).lower():
            print("      fix:  python -c \"import nltk; "
                  "nltk.download('vader_lexicon')\"")
        if "en_core_web_sm" in str(e):
            print("      fix:  python -m spacy download en_core_web_sm")
        if "corpora" in str(e).lower() and "textblob" in str(e).lower():
            print("      fix:  python -m textblob.download_corpora")
        return None


def extract_features_v3(texts, device=-1, blocks=None):
    """Run every available block and join the results."""
    texts = [str(t) if t is not None else "" for t in texts]
    all_blocks = {
        "typography": lambda: typography_features(texts),
        "vader": lambda: vader_features(texts),
        "textblob": lambda: textblob_features(texts),
        "readability": lambda: readability_features(texts),
        "spacy": lambda: spacy_features(texts),
        "emotion": lambda: emotion_features(texts, device=device),
        "irony": lambda: irony_features(texts, device=device),
    }
    want = blocks or list(all_blocks)
    parts, got, missing = [], [], []
    print(f"extracting v3 features for {len(texts)} documents")
    for name in want:
        f = _safe(name, all_blocks[name])
        if f is None:
            missing.append(name)
        else:
            parts.append(f.reset_index(drop=True))
            got.append(name)
    if not parts:
        raise SystemExit("No feature blocks available - install the libraries "
                         "listed at the top of this file.")
    F = pd.concat(parts, axis=1)
    print(f"\n  {F.shape[1]} features from: {', '.join(got)}")
    if missing:
        print(f"  skipped (not installed): {', '.join(missing)}")
    return F
