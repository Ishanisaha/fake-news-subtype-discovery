# Data

No dataset is included in this repository. All sources are public and must be downloaded from their original providers under their own licenses.

## Sources

| Sub-type | Collection 1 | Collection 2 |
|---|---|---|
| Propaganda | Propaganda Techniques Corpus (Da San Martino et al., 2019) | SemEval-2019 Task 4, hyperpartisan news (Kiesel et al., 2019) |
| Rumor | PHEME (Kochkina et al., 2018) | RumourEval-2019 (Gorrell et al., 2019) |
| Satire | News Headlines Dataset for Sarcasm Detection (Misra and Arora, 2023) | OnionOrNot |
| Clickbait | Clickbait Corpus (Chakraborty et al., 2016) | Clickbait detection dataset (Hugging Face, `christinacdl/clickbait_detection_dataset`) |

## Expected files

Place the assembled corpus here as `corpus_raw.csv` (17,465 documents) with these columns:

| Column | Content |
|---|---|
| `text` | document text, original case and punctuation |
| `label` | `propaganda`, `rumour`, `satire` or `clickbait` |
| `collection` | `1` or `2` |
| `src` | source dataset |
| `n_words` | number of words |

`01_clean_corpus.py` writes the cleaned `corpus.csv`, which every other script reads. The assembled corpus is available from the authors on reasonable request, subject to the licenses of the original datasets.

`llm_sample_ids.csv` lists the 1,000 documents (row numbers in `corpus.csv`) used for the zero-shot LLM baseline.

For the external evaluation (`04_cross_validation.py --part external`), put the labelled external pool in `data/external/`:
`eval_balanced_pool.csv`, `labels_local.csv` and `labels_qwen25.csv`.
