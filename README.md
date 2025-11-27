# KGREC Implementation

This is source code of paper : "KGRec: a knowledge graph attention-based model for recommender system"

## Highlights
- Unified training interface across multiple recommenders with shared utilities for logging, evaluation, and checkpointing.
- Dataset-ready directory layout for Amazon-Book, Last-FM, MovieLens-1M, and Yelp2018 (entity lists, KG triples, splits, pretrained MF embeddings).
- Modular loaders (`loaders/`) that encapsulate CF sampling, KG batching, and model-specific preprocessing.
- YAML-driven configs stored under `configs/` so experiments are reproducible and tweakable without code edits.
- Metrics: precision, recall, F1, and NDCG at configurable cutoffs (`Ks`).


## Repository Layout

```
├── configs/                # YAML configs per model and dataset
├── datasets/               # Preprocessed splits and KG triples
├── loaders/                # Data pipelines (base + model-specific)
├── models/                 # Model definitions (KGRec, AMIE, CBFM, …)
├── training/               # Training entry points (one per model)
├── utils/                  # Logging, metrics, and checkpoint helpers
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation (this file)
```


## Prerequisites

- Python 3.10+ (tested on 3.11/3.12/3.13)
- CUDA-capable GPU (optional, but strongly recommended for KG-based models)
- Install dependencies:

```bash
pip install -r requirements.txt
```


## Datasets

The repository expects datasets under `datasets/<dataset-name>/` with the following assets:

- `train.txt` / `test.txt`: user–item interactions (user id followed by purchased/listened item ids).
- `kg_final.txt`: triples `(head, relation, tail)` for KG-aware models.
- `mf.npz`: pretrained MF embeddings (optional, used when `use_pretrain: 1`).
- `entity_list.txt`, `relation_list.txt`, `item_list.txt`, `user_list.txt`: reference mappings.

Place additional datasets in the same format if needed and add matching config files.


## Running Experiments

All training scripts follow the pattern:

```bash
python training/<train_script>.py --configs configs/<Model>/<config>.yaml
```

### Example Commands

- KGRec on Amazon-Book:
	```bash
	python training/train_kgrec.py --configs configs/KGRec/kgrec_amazon_book.yaml
	```
- AMIE (multi-interest CF):
	```bash
	python training/train_amie.py --configs configs/AMIE/amie_amazon_book.yaml
	```
- CBFM (context-based FM, context optional):
	```bash
	python training/train_cbfm.py --configs configs/CBFM/cbfm_amazon_book.yaml
	```
- MCRec baseline:
	```bash
	python training/train_mcrec.py --configs configs/MCRec/mcrec_amazon_book.yaml
	```
- KGAT:
	```bash
	python training/train_kgat.py --configs configs/KGAT/kgat_amazon_book.yaml
	```
- KGNN-LS:
	```bash
	python training/train_kgnn_ls.py --configs configs/KGNN_LS/kgnn_ls_amazon_book.yaml
	```
- CKAN:
	```bash
	python training/train_ckan.py --configs configs/CKAN/ckan_amazon_book.yaml
	```

Logs, checkpoints, metrics CSVs, and runtime summaries are saved under the `save_dir` declared in each config.

