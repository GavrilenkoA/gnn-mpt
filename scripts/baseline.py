import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

# ------------------------- I/O -------------------------
TRAIN_PATH = Path('/disk/10tb/home/gavrilenko/gnn-mpt/data/raw/training_data.csv')
TEST_PATH  = Path('/disk/10tb/home/gavrilenko/gnn-mpt/data/raw/testing_data.csv')

train_df = pd.read_csv(TRAIN_PATH).drop_duplicates()
test_df  = pd.read_csv(TEST_PATH).drop_duplicates()

# убираем пересечение трёх полей из теста, чтобы не дублировать записи
merged_df = test_df.merge(train_df, on=['Antigen', 'HLA', 'CDR3'], how='left', indicator=True)
test_df = test_df[~test_df.index.isin(merged_df[merged_df['_merge'] == 'both'].index)]

# ------------------------- negatives -------------------------
def generate_triplet_negatives(
    df: pd.DataFrame,
    pep_col: str = "Antigen",
    mhc_col: str = "HLA",
    tcr_col: str = "CDR3",
    k: int = 1,
    seed: int = 42,
    max_tries_per_sample: int = 50,
) -> pd.DataFrame:
    """
    На каждый позитив (строку df) для данного pep сэмплируем k пар (HLA, TCR),
    которых не было с этим pep. Возвращаем объединённый df с label={1,0}.
    """
    rng = np.random.default_rng(seed)
    all_mhc = df[mhc_col].unique()
    all_tcr = df[tcr_col].unique()

    seen_pairs_by_pep = {
        pep: set(zip(g[mhc_col].tolist(), g[tcr_col].tolist()))
        for pep, g in df.groupby(pep_col, sort=False)
    }

    neg_rows = []
    for pep, g in df.groupby(pep_col, sort=False):
        n_pos = len(g)
        need = n_pos * k
        seen_pairs = set(seen_pairs_by_pep[pep])
        chosen_pairs = set()
        tries = 0
        while len(chosen_pairs) < need and tries < need * max_tries_per_sample:
            mhc = rng.choice(all_mhc); tcr = rng.choice(all_tcr)
            if (mhc, tcr) not in seen_pairs and (mhc, tcr) not in chosen_pairs:
                chosen_pairs.add((mhc, tcr))
            tries += 1
        for mhc, tcr in chosen_pairs:
            neg_rows.append({pep_col: pep, mhc_col: mhc, tcr_col: tcr})

    pos = df.copy(); pos["label"] = 1
    neg = pd.DataFrame(neg_rows, columns=[pep_col, mhc_col, tcr_col]); neg["label"] = 0
    return pd.concat([pos, neg], ignore_index=True)

train_df_labeled = generate_triplet_negatives(train_df, k=1, seed=42)
test_df_labeled  = generate_triplet_negatives(test_df,  k=1, seed=123)


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def bin_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc":  float(average_precision_score(y_true, y_score)),
    }

@dataclass
class Config:
    seed: int = 42
    val_size: float = 0.2  # доля от train -> valid

    emb_dim: int = 128
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2

    epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda"

    early_patience: int = 6
    ckpt_path: str = "best_triplet.pt"

# ------------------------- graph from DFs -------------------------
class TripletGraph:
    """
    - Строим ID‑карты по (train ∪ test), чтобы индексы теста были валидны.
    - Рёбра графа строим ТОЛЬКО из (train ∪ valid), чтобы не было утечки теста.
    - Пакуем индексы трёх выборок (train/valid/test) в отдельные тензоры.
    """
    def __init__(self, df_train_labeled: pd.DataFrame, df_test_labeled: pd.DataFrame, cfg: Config):
        self.df_tr_all = df_train_labeled.reset_index(drop=True)
        self.df_te_all = df_test_labeled.reset_index(drop=True)
        for df, name in [(self.df_tr_all, "train"), (self.df_te_all, "test")]:
            for c in ["Antigen","HLA","CDR3","label"]:
                if c not in df.columns:
                    raise ValueError(f"{name} dataframe missing column '{c}'")
        self.cfg = cfg
        self.pid = {}; self.mid = {}; self.tid = {}
        self.data: HeteroData = None

    def build_id_maps(self):
        all_p = pd.Index(pd.unique(pd.concat([self.df_tr_all["Antigen"], self.df_te_all["Antigen"]])))
        all_m = pd.Index(pd.unique(pd.concat([self.df_tr_all["HLA"],     self.df_te_all["HLA"]    ])))
        all_t = pd.Index(pd.unique(pd.concat([self.df_tr_all["CDR3"],    self.df_te_all["CDR3"]   ])))
        self.pid = {v:i for i,v in enumerate(all_p)}
        self.mid = {v:i for i,v in enumerate(all_m)}
        self.tid = {v:i for i,v in enumerate(all_t)}

    def _edge_index(self, pairs: list) -> torch.Tensor:
        if len(pairs) == 0:
            return torch.empty(2, 0, dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    def build_graph_and_packs(self) -> HeteroData:
        # split train -> train/valid (стратифицировано)
        tr_idx, va_idx = train_test_split(
            np.arange(len(self.df_tr_all)),
            test_size=self.cfg.val_size, random_state=self.cfg.seed,
            stratify=self.df_tr_all["label"].values
        )
        df_tr = self.df_tr_all.iloc[tr_idx].reset_index(drop=True)
        df_va = self.df_tr_all.iloc[va_idx].reset_index(drop=True)
        df_te = self.df_te_all.reset_index(drop=True)

        # рёбра берём из train ∪ valid
        df_edges = pd.concat([df_tr, df_va], ignore_index=True)

        data = HeteroData()
        data["pep"].num_nodes = len(self.pid)
        data["mhc"].num_nodes = len(self.mid)
        data["tcr"].num_nodes = len(self.tid)

        # P-M edges
        pm_edges = [(self.pid[p], self.mid[m]) for p,m in zip(df_edges["Antigen"], df_edges["HLA"])]
        data["pep","binds","mhc"].edge_index = self._edge_index(pm_edges)

        # M-T edges
        mt_edges = [(self.mid[m], self.tid[t]) for m,t in zip(df_edges["HLA"], df_edges["CDR3"])]
        data["mhc","presents_to","tcr"].edge_index = self._edge_index(mt_edges)

        # (P-T edges removed)

        def pack(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
            return {
                "pep": torch.tensor([self.pid[p] for p in df["Antigen"].values], dtype=torch.long),
                "mhc": torch.tensor([self.mid[m] for m in df["HLA"].values],     dtype=torch.long),
                "tcr": torch.tensor([self.tid[t] for t in df["CDR3"].values],    dtype=torch.long),
                "y":   torch.tensor(df["label"].astype("int64").values),
            }

        data["triplet_train"] = pack(df_tr)
        data["triplet_valid"] = pack(df_va)
        data["triplet_test"]  = pack(df_te)
        self.data = data
        return data


class TripletOnlyGNN(nn.Module):
    def __init__(self, n_pep, n_mhc, n_tcr, emb_dim=128, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.emb = nn.ModuleDict({
            "pep": nn.Embedding(n_pep, emb_dim),
            "mhc": nn.Embedding(n_mhc, emb_dim),
            "tcr": nn.Embedding(n_tcr, emb_dim),
        })
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                HeteroConv({
                    ("pep","binds","mhc"): SAGEConv((-1,-1), hidden),
                    ("mhc","presents_to","tcr"): SAGEConv((-1,-1), hidden),
                }, aggr="mean")
            )
        # приведение всех трёх типов к единому hidden
        self.proj_pep = nn.Linear(emb_dim, hidden)
        self.proj_mhc = nn.Identity()
        self.proj_tcr = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(3*hidden, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def _node_embs(self, data: HeteroData, device) -> Dict[str, torch.Tensor]:
        h = {
            "pep": self.emb["pep"](torch.arange(data["pep"].num_nodes, device=device)),
            "mhc": self.emb["mhc"](torch.arange(data["mhc"].num_nodes, device=device)),
            "tcr": self.emb["tcr"](torch.arange(data["tcr"].num_nodes, device=device)),
        }
        edge_index_dict = {
            ("pep","binds","mhc"): data["pep","binds","mhc"].edge_index,
            ("mhc","presents_to","tcr"): data["mhc","presents_to","tcr"].edge_index,
        }
        for conv in self.layers:
            out = conv(h, edge_index_dict)  # вернёт только dst-типы (mhc, tcr)
            out = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k,v in out.items()}
            # сохраняем представления для типов, которых нет в out (pep)
            h = {k: out.get(k, h[k]) for k in h.keys()}
        return h

    def forward_logits(self, data: HeteroData, device, pack: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self._node_embs(data, device)
        hp = self.proj_pep(h["pep"])[pack["pep"]]
        hm = self.proj_mhc(h["mhc"])[pack["mhc"]]
        ht = self.proj_tcr(h["tcr"])[pack["tcr"]]
        return self.head(torch.cat([hp, hm, ht], dim=-1)).squeeze(-1)

# ------------------------- training -------------------------
class Runner:
    def __init__(self, cfg: Config, data: HeteroData):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.data = data

        # переносим рёбра на девайс
        for et in data.edge_types:
            data[et].edge_index = data[et].edge_index.to(self.device)
        # пакеты индексов
        for split in ["triplet_train","triplet_valid","triplet_test"]:
            for k,v in data[split].items():
                data[split][k] = v.to(self.device)

        self.model = TripletOnlyGNN(
            n_pep=data["pep"].num_nodes,
            n_mhc=data["mhc"].num_nodes,
            n_tcr=data["tcr"].num_nodes,
            emb_dim=cfg.emb_dim, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout
        ).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.crit = nn.BCEWithLogitsLoss()

    def _step(self, split_key: str, train_mode: bool):
        m = self.model; d = self.data
        m.train() if train_mode else m.eval()
        with torch.set_grad_enabled(train_mode):
            logits = m.forward_logits(d, self.device, d[split_key])
            y = d[split_key]["y"].float()
            loss = self.crit(logits, y)
            if train_mode:
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            with torch.no_grad():
                scores = torch.sigmoid(logits).detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                mets = bin_metrics(y_np, scores)
        return float(loss.detach().item()), mets

    def fit_eval(self):
        best_val_pr = -1.0
        best_state = None
        patience = 0

        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss, tr_m = self._step("triplet_train", train_mode=True)
            va_loss, va_m = self._step("triplet_valid", train_mode=False)
            log = {"epoch": epoch, "train_loss": tr_loss, "train": tr_m, "valid_loss": va_loss, "valid": va_m}
            print(json.dumps(log))

            improved = va_m["pr_auc"] > best_val_pr + 1e-6
            if improved:
                best_val_pr = va_m["pr_auc"]
                best_state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                torch.save(best_state, self.cfg.ckpt_path)
                patience = 0
            else:
                patience += 1
                if patience >= self.cfg.early_patience:
                    break

        # тест на лучшей модели
        if best_state is not None:
            self.model.load_state_dict(best_state)
        te_loss, te_m = self._step("triplet_test", train_mode=False)
        print(json.dumps({"best_val_pr_auc": best_val_pr, "test_loss": te_loss, "test": te_m}))
        return {"best_val_pr_auc": best_val_pr, "test": te_m, "ckpt": self.cfg.ckpt_path}


def main():
    cfg = Config()
    set_seed(cfg.seed)

    gb = TripletGraph(train_df_labeled, test_df_labeled, cfg)
    gb.build_id_maps()
    data = gb.build_graph_and_packs()

    runner = Runner(cfg, data)
    res = runner.fit_eval()
    return res


torch.set_float32_matmul_precision("high")
if __name__ == "__main__":
    main()
