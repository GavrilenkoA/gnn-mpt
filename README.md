## gnn-mpt — Graph learning for peptide–MHC–TCR (PMT)

Repository for Public Health Hackathon' 2025 hackathon work on predicting ternary interactions between peptide (Antigen), MHC (HLA) and TCR (CDR3) using heterogeneous Graph Neural Networks (GNNs) and a simple MLP baseline.

The codebase contains several training pipelines ranging from simple GNN baselines with random node embeddings to GAT-based models leveraging precomputed protein language model (pLM) features, as well as multi-task and two-stage training variants.


### Quick start

1) Create environment and install dependencies

```bash
conda create -n unipmt python=3.11 -y
conda activate unipmt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

2) Download data and place it according to Data layout, or adjust paths in scripts:

- Data download: [Google Drive folder](https://drive.google.com/drive/folders/17ht6OBhv34LrZBm9Y2ow_A3KkuAk8tmG)
- Then adjust hardcoded paths inside the target script if needed.

3) Run a model, e.g. a GAT-based triplet classifier:

```bash
python scripts/gat.py
```

Training logs are printed as JSON per epoch; the best checkpoint is saved to the path specified in the script config.


### Repository structure

```
gnn-mpt/
├─ scripts/
│ ├─ gnn.py # SAGEConv baseline on (pep–mhc, mhc–tcr) graph with random nn.Embedding
│ ├─ gat.py # GATv2 baseline on the same graph with random nn.Embedding
│ ├─ gat_emb.py # GATv2 using precomputed pLM embeddings (.npz banks) for pep/mhc/tcr
│ ├─ gat2stage.py # Two-stage training: pairs (PM/PT) → triplets (PMT)
│ ├─ mlp.py # Triplet MLP baseline without message passing
│ └─ pmt_meta_learn.py # Multi-task training with PM, PT (BCE) and PMT (InfoNCE/DMF)
│
├─ esmc_embeddings/
│ ├─ calc_esmc_embeddings.py # Compute sequence embeddings for pep/mhc/tcr using an ESM-C model
│ └─ README.md
│
├─ peptides_descriptions_embeddings/
│ ├─ get_descriptions.py # Fetch peptide descriptions via IEDB API
│ ├─ make_embeddings.py # Embed descriptions; produces .npy bank
│ └─ README.md
│
├─ data/ # (Optional) Local data drop-in; not used directly by scripts
├─ requirements.txt
└─ README.md
```


### Data layout

Most training scripts expect CSV files with these columns:

- training_data.csv: Antigen, HLA, CDR3
- testing_data.csv: Antigen, HLA, CDR3

Some pipelines also use pair-only edge CSVs:

- edges_pm_seq_filt.csv: Antigen, HLA
- edges_pt_seq_filt.csv: Antigen, CDR3

By default, paths are hardcoded inside scripts to the following locations; edit them to match your environment:

```
/mnt/nfs_protein/gavrilenko/mpt/raw/training_data.csv
/mnt/nfs_protein/gavrilenko/mpt/raw/testing_data.csv
/mnt/nfs_protein/gavrilenko/mpt/raw/edges_pm_seq_filt.csv
/mnt/nfs_protein/gavrilenko/mpt/raw/edges_pt_seq_filt.csv
```

Triplet datasets are de-duplicated, and exact duplicates across train/test (Antigen, HLA, CDR3) are removed from test inside the scripts.

Data download

- You can pull the prepared datasets from the shared Google Drive folder: [Google Drive folder](https://drive.google.com/drive/folders/17ht6OBhv34LrZBm9Y2ow_A3KkuAk8tmG). Place the CSVs under your local paths (or update the constants in scripts).


### Negative sampling and splits

- Triplet negatives: for each positive triplet (Antigen, HLA, CDR3) we sample k random mismatched pairs (HLA, CDR3) for the same peptide that were not observed with this peptide. The label column `label ∈ {0,1}` is added.
- Pair negatives (PM/PT): random mismatches on the right node with filtering against any known positives of that pair type (including those implied by triplets).
- Train/valid split: stratified on `label` when labels are present; otherwise random split.
- Edges used for message passing are built only from train∪valid to avoid any test leakage. ID maps are created across train∪test so test indices remain valid.


### Models and training scripts

All training scripts share common ideas:

- Build a heterogeneous graph with node types: pep (Antigen), mhc (HLA), tcr (CDR3)
- Edge types used by different pipelines: pep–mhc (binds), mhc–tcr (presents_to), pep–tcr (contacts; only in multi-task/meta variants)
- Evaluate with PR-AUC and ROC-AUC; logs are printed per epoch in JSON
- Checkpoints of the best validation PR-AUC are saved to a path set by `ckpt_*` in each script
- Determinism: `set_seed` is called from `main()`

Models

- scripts/gnn.py
  - Architecture: random `nn.Embedding` for each node type, HeteroConv with SAGEConv on (pep–mhc) and (mhc–tcr), MLP head over concatenated (pep, mhc, tcr) states
  - Negatives: triplet negatives (k=1 by default)
  - Metrics: PR-AUC, ROC-AUC; early stopping by validation PR-AUC

- scripts/gat.py
  - Architecture: random `nn.Embedding`, HeteroConv with `GATv2Conv` (concat=False, no self-loops), residual+ELU+Dropout, MLP head
  - Typically uses k=2 negatives for train in the shipped config

- scripts/mlp.py
  - No graph message passing; learns random embeddings per node type and classifies concatenated (pep, mhc, tcr) embeddings using a multilayer perceptron
  - Fast baseline for sanity checks; follows same negative sampling and split policy

Using precomputed embeddings

- scripts/gat_emb.py
  - Instead of `nn.Embedding`, the graph packs per-node features into `data[ntype].x` using precomputed banks
  - Each node-type feature is projected to a common `hidden` and passed through a stack of `GATv2Conv`
  - Required .npz banks (see Embedding banks below):
    - `/mnt/nfs_protein/gavrilenko/mpt/raw/peptides_data.npz`
    - `/mnt/nfs_protein/gavrilenko/mpt/raw/tcr_data.npz`
    - `/mnt/nfs_protein/gavrilenko/mpt/raw/mhc_data.npz`

Two-stage training (pairs → triplets)

- scripts/gat2stage.py
  - Stage 1 trains pair heads on PM and PT with BCE
  - Stage 2 trains triplet head on PMT (optionally with multitask regularization from pair heads)
  - Graph edges include pep–mhc, pep–tcr, mhc–tcr, all formed from train∪valid pairs
  - Negatives are generated with guards against any known positives of the corresponding type

Multi-task with InfoNCE/DMF

- scripts/pmt_meta_learn.py
  - Graph adds reverse edges and a direct pep–tcr edge
  - Three objectives: PM (BCE), PT (BCE with approximate marginalization over sampled M), PMT (InfoNCE using DMF-style hp⊙hm/ht interactions and in-batch negatives)
  - Validation is reported on the triplet task; test is on triplets only


### Embeddings: how to create and use

ESM-C sequence embeddings

- `esmc_embeddings/calc_esmc_embeddings.py` computes sequence embeddings for all three node types using an ESM-C model.
  - Inputs required:
    - `pseudoseqs.csv` with column `mhc` and one-hot columns `{position}_{aminoacid}` for MHC pseudosequences
    - `nodes_peptides.csv` with column `sequence`
    - `nodes_tcr.csv` with column `sequence`
  - Authentication: a Hugging Face token with write permissions is required by the script (see that file for details).

Peptide description text embeddings

- `peptides_descriptions_embeddings/get_descriptions.py` fetches textual descriptions for unique peptide sequences via the IEDB API.
- `peptides_descriptions_embeddings/make_embeddings.py` embeds those texts and writes:
  - `unique_peptides_embeddings.npy`
  - `unique_peptides_sequences.csv` (sequences in the same order as embeddings)

Embedding banks (.npz) expected by `gat_emb.py`

`scripts/gat_emb.py` accepts two .npz formats via `load_bank()`:

- Matrix format
  - Arrays: `keys` (or `ids`) with string identifiers, and `embs` (or `embeddings`) of shape [N, D]
- Per-key format
  - Multiple arrays whose names are actual identifiers; each value is a vector of shape [D]

At runtime, the script extracts a subset matching the vocabulary seen in train∪test and inserts it into `data[ntype].x` for each node type.


### How to run

All training scripts are standalone. Run them directly and adjust `Config` (a `@dataclass` inside each script) if needed. Important fields:

- `device`: e.g. `cuda:0`, `cuda:1`, or `cpu`
- `epochs`, `lr`, `weight_decay`, `dropout`, `layers`, `hidden`, `heads`
- `ckpt_path` / `ckpt_pairs` / `ckpt_triplet`
- Negative sampling knobs (k, ratios) differ per script

Examples

```bash
# SAGE baseline
python scripts/gnn.py

# GAT baseline
python scripts/gat.py

# GAT with precomputed embeddings
python scripts/gat_emb.py

# Two-stage pairs→triplets
python scripts/gat2stage.py

# Multi-task (PM, PT, PMT)
python scripts/pmt_meta_learn.py

# MLP baseline (no message passing)
python scripts/mlp.py
```


### Logging and checkpoints

- Per-epoch logs are printed as JSON (one line per epoch) with training/validation losses and metrics.
- Early stopping is applied based on validation PR-AUC.
- The best model state dict is saved to the configured checkpoint path. Final test metrics are printed at the end.


### Reproducibility

Each script calls `set_seed(cfg.seed)` which seeds Python, NumPy, and PyTorch (CPU and CUDA). For full determinism you may need to set deterministic flags in PyTorch and control cuDNN; the current setup prioritizes performance while reducing variance.


### Requirements

See `requirements.txt` for fully pinned versions. Key libraries:

- PyTorch 2.8.0 + CUDA 12.6
- torch-geometric 2.6.1
- scikit-learn, pandas, numpy
- transformers (for text/ESM utilities)


### Troubleshooting

- Data paths: scripts use hardcoded `Path(...)` constants near the top. Change them to your local paths.
- Device selection: if you do not have a GPU, set `device = "cpu"` in the `Config` of the script you run.
- Empty or missing embeddings when using `gat_emb.py`: the loader prints how many keys were missing. Ensure your .npz banks include all identifiers present in train∪test.
- Class imbalance: PR-AUC is the primary metric reported; adjust negative sampling ratios if needed for your experiments.


### Acknowledgements

Developed for the Public Health Hackathon' 2025. Uses PyTorch Geometric for heterogeneous graph modeling and standard pLMs for protein/TCR/MHC representations.
