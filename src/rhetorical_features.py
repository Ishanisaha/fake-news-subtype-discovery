"""
Text cleaning for the semantic stream and the 13 rhetorical features.

  f_sent_len     average sentence length          f_modal        modal / uncertainty verbs
  f_ttr          type-token ratio                 f_attribution  vague source attribution
  f_emotion      emotional intensity              f_exclamation  exclamation density
  f_hedging      hedging expressions              f_rhet_q       rhetorical questions
  f_question     question marks                   f_satire_mk    satirical register markers
  f_direct_addr  direct address (you / your)      f_listicle     sentences opening with a number
                                                  f_forward      sentences opening with a forward reference

All features are computed on the raw text (case and punctuation kept).
"""
import re

import numpy as np
import pandas as pd

_EVENT_PATS = [
    r"\bcharliehebdo\b", r"\bcharlie\s*hebdo\b",
    r"\bferguson\b",     r"\bmikebrown\b",
    r"\bsydneysiege\b",  r"\bsydney\s*siege\b",
    r"\bottawashooting\b", r"\botto?awa\s*shooting\b",
    r"\bgermanwings\b",  r"\bparisshooting\b",
]

def clean_text_aggressive(text: str) -> str:
    """Remove URLs, retweet markers, CamelCase hashtags and event names; lowercase; keep letters only."""
    t = str(text)
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"\bRT\b", " ", t)
    t = re.sub(r"\binch\b", "in", t, flags=re.IGNORECASE)
    t = re.sub(r"\bangstrom\b", " ", t, flags=re.IGNORECASE)
    # CamelCase hashtag tokens (OttawaShooting, SydneySiege …)
    t = re.sub(r"\b[A-Z][a-z]+[A-Z][a-z]\w*\b", " ", t)
    for pat in _EVENT_PATS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    t = t.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


HEDGING_TERMS = [
    # Epistemic modals
    "might","may","could","possibly","probably","perhaps","maybe","presumably",
    "apparently","seemingly","reportedly","allegedly","supposedly","purportedly",
    "ostensibly","evidently","arguably","conceivably",
    # Multi-word evidential phrases
    "sources say","sources claim","sources report","according to sources",
    "unnamed sources","anonymous sources","officials say","officials claim",
    "witnesses say","witnesses report","experts say","experts believe",
    "reports suggest","reports indicate","it has been reported","it is reported",
    "it is claimed","it is believed","it is understood","it is said",
    "word has it","rumour has it","some claim","some believe","many believe",
    "people are saying","there are reports","unconfirmed reports",
    "unverified claims","unverified reports",
    # Uncertainty markers
    "unclear","uncertain","unconfirmed","unverified","unsubstantiated",
    "not confirmed","not yet confirmed","not yet verified",
    "alleged","purported","claimed","suspected",
    "appears to","seems to","looks like",
    # Spreading signals (Zubiaga 2016)
    "breaking","just in","developing","please share","share this",
    "urgent","warning","alert","stay tuned","more to follow",
    "details emerging","updates to follow","situation developing",
]

# Source: Mohammad & Turney (2013), NRC Word-Emotion Association Lexicon
# Subset: fear, anger, disgust, anticipation — most relevant to fake news
EMOTION_TERMS = [
    # Fear / threat
    "threat","danger","risk","crisis","emergency","attack","violence",
    "terror","terrorist","bomb","weapon","kill","death","murder","war",
    "invasion","catastrophe","disaster","collapse","destruction","chaos",
    "panic","horror","nightmare","dread","peril",
    # Anger / outrage
    "outrage","outraged","furious","fury","rage","angry","anger",
    "disgust","disgusting","appalling","shocking","scandalous","scandal",
    "corrupt","corruption","betrayal","betrayed","fraud","fraudulent",
    "criminal","crime",
    # Anticipation / urgency (clickbait signals)
    "incredible","amazing","unbelievable","stunning","astonishing",
    "extraordinary","unprecedented","historic","explosive","bombshell",
    "revelation","revealed","exposed",
    # Sadness / sympathy (propaganda and rumour)
    "tragic","tragedy","devastating","heartbreaking","victims","suffering",
    "innocent","helpless","desperate","dire","grim","bleak",
]

# Satire structural register patterns
SAT_STRUCT_PATS = [
    r"\barea\s+man\b", r"\barea\s+woman\b", r"\barea\s+resident\b",
    r"\bexperts?\s+say\b", r"\bstudy\s+finds\b", r"\bnew\s+report\b",
    r"\bsources?\s+confirm\b", r"\blocal\s+man\b", r"\blocal\s+woman\b",
    r"\baccording\s+to\s+a\s+new\b",
]

# Vague attribution patterns
ATTR_PATS_LIST = [
    "sources say","sources claim","witnesses say","it has been reported",
    "some claim","people are saying","word has it","there are reports",
    "it is claimed","it is believed","according to sources",
]


def _eleven_features(df: pd.DataFrame) -> pd.DataFrame:
    """Eleven base rhetorical features, computed on the raw text."""
    def _count_hedging(raw: str) -> int:
        """Count hedging/uncertainty expressions using 123-term lexicon."""
        text_lower = str(raw).lower()
        score = 0
        for term in HEDGING_TERMS:
            if ' ' in term:
                if term in text_lower:
                    score += 1
            else:
                if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                    score += 1
        return score

    def _emotion_score(raw: str) -> float:
        """Emotion intensity using NRC subset + typographic signals."""
        s      = str(raw)
        ws     = s.split()
        excl   = s.count('!')
        ques   = s.count('?')
        caps   = sum(1 for w in ws if w.isupper() and len(w) > 2)
        emo_w  = sum(1 for w in EMOTION_TERMS
                     if re.search(r'\b' + re.escape(w) + r'\b', s.lower()))
        return excl * 0.5 + ques * 0.3 + caps * 0.8 + emo_w

    def _sent_len(raw: str) -> float:
        """Avg words per sentence — uses raw punctuation as boundaries."""
        ss = [s.strip() for s in re.split(r'[.!?]+', str(raw)) if s.strip()]
        return float(np.mean([len(s.split()) for s in ss])) if ss else 0.0

    def _ttr(raw: str) -> float:
        ws = str(raw).split()
        return len(set(w.lower() for w in ws)) / max(len(ws), 1)

    def _direct_addr(raw: str) -> float:
        ws = str(raw).lower().split()
        return sum(1 for w in ws if w in ["you", "your", "yourself"]) / max(len(ws), 1)

    def _modal(raw: str) -> int:
        return len(re.findall(
            r'\b(could|might|may|possibly|perhaps|apparently|seems?|appears?)\b',
            str(raw).lower()))

    def _attribution(raw: str) -> int:
        text_lower = str(raw).lower()
        return sum(1 for t in ATTR_PATS_LIST if t in text_lower)

    def _rhet_q(raw: str) -> int:
        return len(re.findall(
            r'\b(?:can|do|are|would|should|what|why|how|which|who)\b[^.!?]*\?',
            str(raw).lower()))

    def _satire_mk(raw: str) -> int:
        text_lower = str(raw).lower()
        return sum(1 for p in SAT_STRUCT_PATS if re.search(p, text_lower))

    raw = df["text"].fillna("")

    df["f_sent_len"]    = raw.apply(_sent_len)
    df["f_ttr"]         = raw.apply(_ttr)
    df["f_emotion"]     = raw.apply(_emotion_score)
    df["f_hedging"]     = raw.apply(_count_hedging)
    df["f_question"]    = raw.apply(lambda t: str(t).count('?'))
    df["f_direct_addr"] = raw.apply(_direct_addr)
    df["f_modal"]       = raw.apply(_modal)
    df["f_attribution"] = raw.apply(_attribution)
    df["f_exclamation"] = raw.apply(lambda t: str(t).count('!') / max(len(str(t)), 1) * 100)
    df["f_rhet_q"]      = raw.apply(_rhet_q)
    df["f_satire_mk"]   = raw.apply(_satire_mk)
    return df


NUMW = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fifteen|twenty|thirty|fifty|hundred)"
FWD = ("this", "these", "here", "here's", "heres", "why", "what", "how", "you", "your", "which", "who",
       "when", "can", "do", "will", "is", "are", "should")


def _sentences(t):
    ss = [s.strip().lower() for s in re.split(r'(?<=[.!?])\s+', str(t)) if len(s.strip().split()) >= 2]
    return ss or [str(t).lower()]


def _listicle(t):
    return float(np.mean([bool(re.match(NUMW + r"\b", s)) for s in _sentences(t)]))


def _forward(t):
    return float(np.mean([(s.split() or [""])[0].strip('"\'‘“') in FWD for s in _sentences(t)]))


def extract_rhetorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _eleven_features(df)
    raw = df["text"].fillna("")
    df["f_listicle"] = raw.apply(_listicle)
    df["f_forward"] = raw.apply(_forward)
    return df
