"""
Remove source markers from the assembled corpus.

Markers that identify the source dataset rather than the writing are removed:
HTML entities are decoded, URLs are removed, the [SEP] separator between
headlines becomes a sentence boundary, the Onion site-name tail is removed and
a leading @ or # is dropped from words. Case, punctuation and document length
are kept.

Input : data/corpus_raw.csv
Output: data/corpus.csv, results/cleaning_report.csv
"""
import html, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RES = os.path.join(HERE, '..', 'results')
SRC = os.path.join(DATA, 'corpus_raw.csv')
OUT = os.path.join(DATA, 'corpus.csv')

ONION = re.compile(r"\s*\|?\s*the onion\s*-\s*america['’]s finest news source\.?", re.I)
URL = re.compile(r'https?://\S+|www\.\S+')
SEP = re.compile(r'\s*\[SEP\]\s*')
TAG = re.compile(r'(?<![\w@#])[@#](?=\w)')
MARKERS = {'sep_token': r'\[SEP\]', 'html_entity': r'&#\d+;|&[a-z]+;', 'url': r'https?://|www\.',
           'onion_site_name': r"america['’]s finest news source", 'at_or_hash': r'(?<![\w&])[@#]\w'}


def clean(t):
    t = str(t)
    for _ in range(2):
        t = html.unescape(t)
    t = URL.sub(' ', t)
    t = ONION.sub(' ', t)
    t = SEP.sub(' [[SEP]] ', t)
    t = re.sub(r'([.?!:;…])\s*\[\[SEP\]\]', r'\1', t)
    t = t.replace(' [[SEP]] ', '. ').replace('[[SEP]]', '. ')
    t = TAG.sub('', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def count(df):
    return {k: df.text.str.contains(p, regex=True, case=False).groupby([df.label, df.src]).sum() for k, p in MARKERS.items()}


def main():
    if not os.path.exists(SRC):
        sys.exit('corpus_raw.csv not found in ' + os.path.abspath(DATA) + ' (see data/README.md)')
    df = pd.read_csv(SRC)
    before = count(df)
    df['text'] = df['text'].map(clean)
    df['n_words'] = df['text'].str.split().str.len()
    after = count(df)
    rep = pd.concat({f'{k}_before': before[k] for k in MARKERS} | {f'{k}_after': after[k] for k in MARKERS}, axis=1)
    os.makedirs(RES, exist_ok=True)
    rep.to_csv(os.path.join(RES, 'cleaning_report.csv'))
    df.to_csv(OUT, index=False)
    print(rep.to_string())
    print(f'\n{len(df):,} documents written to {OUT}')
    left = rep[[c for c in rep.columns if c.endswith('_after')]].to_numpy().sum()
    print('markers left:', int(left))


if __name__ == '__main__':
    main()
