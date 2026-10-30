"""
Zero-shot LLM baseline (Qwen2.5-7B and Llama-3.1-8B through Ollama, temperature 0,
seed 42) on 1,000 documents (250 per sub-type, listed in data/llm_sample_ids.csv).

    ollama pull qwen2.5:7b
    ollama pull llama3.1:8b
    python 11_llm_baseline.py --benchmark     # time five documents first
    python 11_llm_baseline.py

Progress is saved every 25 documents and the run resumes if stopped.
Output: results/llm_baseline/
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, 'data', 'corpus.csv')
IDS = os.path.join(ROOT, 'data', 'llm_sample_ids.csv')
OUT = os.path.join(ROOT, 'results', 'llm_baseline')
URL = 'http://localhost:11434'
WORDS, TEMPERATURE, SEED = 150, 0.0, 42
VALID = {'propaganda', 'rumour', 'satire', 'clickbait', 'none', 'unclear'}
C4 = ['propaganda', 'rumour', 'satire', 'clickbait']

SYSTEM_PROMPT = "You are annotating news text for an academic study.\n\nEvery text you see has already been established as false or misleading. You are NOT judging whether it is true. You are judging why it appears to have been written.\n\nAssign exactly one category.\n\npropaganda - Written to advance a political, ideological or institutional agenda. The author wants the reader to adopt a position or take a side. Presents a contested matter as settled.\n\nrumour - An unverified claim circulating during an uncertain or developing situation. The author is passing on something they do not themselves know to be true. The intent is to transmit, not to persuade.\n\nsatire - Humour or irony commenting on real events through exaggeration or absurdity. The author does not expect to be believed; the falsehood is a joke the reader is meant to recognise.\n\nclickbait - Written to make the reader click or keep watching. Information is withheld, teased or overstated so the reader must engage to find out what happened. The motive is attention, not ideology.\n\nnone - It is clearly false, and you can tell why it was written, but the reason is none of the four above. Plain fabrication or a hoax with no persuasive, humorous or attention-seeking purpose belongs here.\n\nunclear - You genuinely cannot tell. Too short, too garbled, or two categories fit equally well.\n\nJudge intent, not topic. A story about an election is not automatically propaganda. A story about a celebrity is not automatically clickbait. Judge only the text given; do not imagine a fuller article and do not look the story up.\n\n'none' and 'unclear' are real answers, not failures. Use them when warranted. But do not use them to avoid deciding when one category clearly fits best.\n\nRespond with JSON only:\n{\"label\": \"<category>\", \"confidence\": <1-5>, \"reason\": \"<one short sentence>\"}"


def check_ollama(models):
    try:
        tags = json.loads(urllib.request.urlopen(URL + '/api/tags', timeout=5).read().decode())
    except Exception:
        sys.exit('Ollama is not running. Open the Ollama app (or run "ollama serve" in another window), then try again.')
    have = {m['name'] for m in tags.get('models', [])}
    missing = [m for m in models if m not in have and m + ':latest' not in have]
    if missing:
        sys.exit('These models are not installed: ' + ', '.join(missing) +
                 '\nInstall with:  ' + '   '.join('ollama pull ' + m for m in missing) +
                 '\nInstalled now: ' + ', '.join(sorted(have)))


def call(model, text):
    req = urllib.request.Request(URL + '/api/chat', data=json.dumps({
        'model': model, 'format': 'json', 'stream': False,
        'options': {'temperature': TEMPERATURE, 'seed': SEED, 'num_predict': 200},
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                     {'role': 'user', 'content': f'TEXT:\n"""\n{text}\n"""\n\nJSON:'}]}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        c = json.loads(r.read().decode())['message']['content']
    try:
        o = json.loads(c)
    except json.JSONDecodeError:
        s, e = c.find('{'), c.rfind('}')
        o = json.loads(c[s:e + 1])
    lab = str(o.get('label', 'unclear')).strip().lower()
    lab = 'rumour' if lab == 'rumor' else lab
    lab = 'none' if lab in ('none of the above', 'other', 'fabrication', 'hoax') else lab
    return (lab if lab in VALID else 'unclear'), str(o.get('reason', ''))[:200]


def measures(g, p):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
    g, p = np.asarray(g), np.asarray(p)
    lt, lp = sorted(set(g)), sorted(set(p))
    cm = np.array([[np.sum((g == a) & (p == b)) for b in lp] for a in lt])
    r, c = linear_sum_assignment(-cm)
    return dict(N=len(g), ARI=round(adjusted_rand_score(g, p), 4), NMI=round(normalized_mutual_info_score(g, p), 4),
                Purity=round(cm.max(axis=0).sum() / len(g), 4), Match=round(cm[r, c].sum() / len(g) * 100, 2),
                Accuracy=round(np.mean(g == p) * 100, 2),
                MacroF1=round(f1_score(g, p, labels=C4, average='macro', zero_division=0), 4),
                none_or_unclear_pct=round(np.isin(p, ['none', 'unclear']).mean() * 100, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['qwen2.5:7b', 'llama3.1:8b'])
    ap.add_argument('--per-class', type=int, default=250)
    ap.add_argument('--benchmark', action='store_true')
    a = ap.parse_args()
    check_ollama(a.models)

    os.makedirs(OUT, exist_ok=True)
    corpus = pd.read_csv(CORPUS)
    df = pd.read_csv(IDS)
    df['text'] = corpus.loc[df['doc_id'], 'text'].to_numpy()
    assert (corpus.loc[df['doc_id'], 'label'].to_numpy() == df['label'].to_numpy()).all(), 'corpus does not match the sample ids'
    df = df.groupby('label', sort=False).head(a.per_class).reset_index(drop=True)

    if a.benchmark:
        for model in a.models:
            t0 = time.time()
            for _, r in df.head(5).iterrows():
                lab, _ = call(model, ' '.join(str(r['text']).split()[:WORDS]))
                print(f'  {model}: gold={r["label"]:<10} answer={lab}')
            sec = (time.time() - t0) / 5
            print(f'{model}: {sec:.1f} s per document -> full run of {len(df)} documents '
                  f'about {sec * len(df) / 3600:.1f} hours\n')
        return

    for model in a.models:
        tag = model.split(':')[0].replace('.', '')
        out = os.path.join(OUT, f'zeroshot_{tag}.csv')
        rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
        done = {r['doc_id'] for r in rows}
        todo = df[~df['doc_id'].isin(done)]
        print(f'\n{model}: {len(rows)} already done, {len(todo)} to go')
        t0 = time.time()
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            text = ' '.join(str(r['text']).split()[:WORDS])
            for attempt in range(3):
                try:
                    lab, why = call(model, text)
                    rows.append(dict(doc_id=r['doc_id'], gold=r['label'], pred=lab, reason=why)); break
                except Exception as e:
                    if attempt == 2:
                        rows.append(dict(doc_id=r['doc_id'], gold=r['label'], pred='unclear', reason='ERROR ' + str(e)[:80]))
                    time.sleep(3)
            if i % 25 == 0 or i == len(todo):
                pd.DataFrame(rows).to_csv(out, index=False)
                rate = (time.time() - t0) / i
                print(f'  {len(rows)}/{len(df)}   {rate:.1f} s/doc   about {(len(todo) - i) * rate / 60:.0f} min left')
        pd.DataFrame(rows).to_csv(out, index=False)

    res = {}
    for model in a.models:
        tag = model.split(':')[0].replace('.', '')
        d = pd.read_csv(os.path.join(OUT, f'zeroshot_{tag}.csv'))
        d = d[d['doc_id'].isin(df['doc_id'])]
        res[model] = measures(d['gold'], d['pred'])
        print(f'\n=== {model} ===')
        print(pd.crosstab(d['gold'], d['pred']).to_string())
    table = pd.DataFrame(res).T
    table.to_csv(os.path.join(OUT, 'zeroshot_summary.csv'))
    print('\n\nSummary:')
    print(table.to_string())


if __name__ == '__main__':
    main()
