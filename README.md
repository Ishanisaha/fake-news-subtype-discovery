# Fake news sub-type discovery

Code for the paper

> I. Saha and S. Puthran, "Beyond True or False: A Two-Stream Framework for Unsupervised Fake News Sub-Type Discovery," submitted to *IEEE Access*.

The framework groups fake news documents into four sub-types (propaganda, rumor, satire and clickbait) without using sub-type labels. Labels are used only at the end, to score the clusters.

## Method

1. **Semantic stream**: Sentence-BERT (`all-MiniLM-L6-v2`) embeddings, L2-normalized.
2. **Rhetorical stream**: 13 hand-crafted rhetorical features (`src/rhetorical_features.py`), min-max scaled and expanded with random Fourier features (D = 200, gamma = 1).
3. **Fusion**: the two streams are weighted 1.5 : 1, concatenated and L2-normalized.
4. **Reduction**: truncated SVD (100) followed by UMAP (40 dimensions, cosine, 15 neighbors, min_dist 0).
5. **Clustering**: K-Means with K = 4.
6. **Consensus**: steps 2 to 5 are repeated for 10 random Fourier draws (seeds 42 to 51). The ten partitions are one-hot encoded, concatenated and clustered again with K-Means (K = 4).
7. **Evaluation**: Hungarian matching to the sub-type labels; ARI, NMI, purity and match rate.

## Installation

Python 3.10 or later.

```
git clone https://github.com/Ishanisaha/fake-news-subtype-discovery.git
cd fake-news-subtype-discovery
pip install -r requirements.txt
```

The library features used in the feature analysis (`02_feature_anova.py`) need a few extra packages:

```
pip install -r requirements-extra.txt
python -m spacy download en_core_web_sm
```

## Data

The datasets are not included here; they have to be downloaded from their original providers under their own licenses. See [`data/README.md`](data/README.md) for the sources and for the file format the scripts expect.

## Reproducing the results

Run the scripts from the `src` folder in this order. All results are written to `results/`; embeddings and UMAP manifolds are cached in `data/` and `results/*/cache/`, so a script can be stopped and started again.

| Script | What it produces |
|---|---|
| `01_clean_corpus.py` | Removes source markers from the corpus |
| `02_feature_anova.py` | Feature effect sizes (Tables VI, VII and XIV) |
| `03_main_tables.py` | Ablation, weight sweep, confusion matrix and choice of K over three seeds |
| `04_cross_validation.py --part all` | Five-fold, cross-collection and external evaluation |
| `05_baselines.py` | Comparison methods (Table XIII) |
| `06_weight_holdout.py` | Held-out choice of the stream weighting |
| `07_sensitivity.py` | Structure for K = 2 to 10; sensitivity to the kernel and UMAP settings |
| `08_draw_stability.py` | 20 random Fourier draws, stability of the consensus, variant without the kernel |
| `09_consensus_tables.py` | Ablation, weight sweep, Gaussian mixture and cross-validation with the consensus step |
| `10_consensus_numbers.py` | Final table values (run again with `boot` for the bootstrap intervals) |
| `11_llm_baseline.py` | Zero-shot LLM baseline (needs [Ollama](https://ollama.com)) |
| `12_figures.py` | Figures |
| `13_controls.py` | Source and length controls (without the hyperpartisan source; length-matched samples) |

`framework.py` holds the shared pipeline code and can also be run on its own for a single end-to-end run.

Random seeds are fixed in every random step (Fourier projection, SVD, UMAP, clustering and data splits). On a laptop CPU, the full sequence takes several hours; most of the time goes into the UMAP fits of scripts 08 and 09.

## Repository layout

```
src/        all code
data/       input data (not tracked) and the document ids of the LLM sample
results/    output tables and caches (created when the scripts run, not tracked)
figures/    output figures (created by 12_figures.py, not tracked)
```

## Citation

If you use this code, please cite the paper above. The citation will be updated once the paper is published.

## License

MIT, see [LICENSE](LICENSE). The license covers the code only, not the datasets.
