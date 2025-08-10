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
from torch_geometric.nn import HeteroConv, GATv2Conv

# ------------------------- I/O -------------------------
TRAIN_PATH = Path('/home/team/data/pmt_pmt/raw/training_data.csv')
TEST_PATH = Path('/home/team/data/pmt_pmt/raw/testing_data.csv')

train_df = pd.read_csv(TRAIN_PATH).drop_duplicates()
test_df  = pd.read_csv(TEST_PATH).drop_duplicates()

# убираем пересечение трёх полей из теста, чтобы не дублировать записи
merged_df = test_df.merge(train_df, on=['Antigen', 'HLA', 'CDR3'], how='left', indicator=True)
test_df = test_df[~test_df.index.isin(merged_df[merged_df['_merge'] == 'both'].index)]


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
test_df_labeled = generate_triplet_negatives(test_df, k=1, seed=123)


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
    lr: float = 2e-4
    weight_decay: float = 1e-4
    device: str = "cuda"

    early_patience: int = 15
    ckpt_path: str = "best_triplet.pt"


class EmbBank:
    """Универсальный банк эмбеддингов: поддерживает два формата .npz."""
    def __init__(self, mode, npz_obj=None, keys=None, embs=None):
        # mode: "perkey" | "matrix"
        self.mode = mode
        self._z = npz_obj       # NpzFile для perkey
        self._pos = None        # dict: key->row (для matrix)
        self._keys = None
        self._embs = None

        if mode == "matrix":
            assert keys is not None and embs is not None
            self._keys = np.asarray(keys, dtype=object)
            self._embs = np.asarray(embs, dtype=np.float32)
            self._pos = {str(k): i for i, k in enumerate(self._keys)}
            self._dim = int(self._embs.shape[1])
        elif mode == "perkey":
            assert npz_obj is not None
            # возьмём размерность из первого элемента
            first = npz_obj.files[0]
            self._dim = int(npz_obj[first].shape[-1])
        else:
            raise ValueError("Unknown mode")

    @property
    def dim(self) -> int:
        return self._dim

    def subset_by_vocab(self, vocab: Dict[str, int], missing: str = "zeros") -> torch.Tensor:
        D = self.dim
        if missing == "zeros":
            out = np.zeros((len(vocab), D), dtype=np.float32)
        else:
            out = np.random.normal(0, 0.02, size=(len(vocab), D)).astype(np.float32)

        miss = 0
        if self.mode == "matrix":
            for k, idx in vocab.items():
                j = self._pos.get(str(k))
                if j is not None:
                    out[idx] = self._embs[j]
                else:
                    miss += 1
        else:  # perkey
            z = self._z
            for k, idx in vocab.items():
                key = str(k)
                if key in z.files:
                    vec = z[key]
                    # защита от типа/формы
                    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                    if vec.shape[0] != D:
                        raise ValueError(f"{key}: dim {vec.shape[0]} != {D}")
                    out[idx] = vec
                else:
                    miss += 1

        if miss:
            print(f"[EmbBank] Missing {miss}/{len(vocab)} keys")
        return torch.from_numpy(out)


def load_bank(npz_path: str) -> EmbBank:
    z = np.load(npz_path, allow_pickle=True)
    names = set(z.files)
    # формат "матрица"
    has_matrix = ({"keys", "embs"} <= names) or ({"ids", "embeddings"} <= names)
    if has_matrix:
        keys = z["keys"] if "keys" in names else z["ids"]
        embs = z["embs"] if "embs" in names else z["embeddings"]
        # приведение типов
        keys = np.array([str(x) for x in keys], dtype=object)
        embs = np.asarray(embs, dtype=np.float32)
        return EmbBank(mode="matrix", keys=keys, embs=embs)
    # формат "per-key": много имён, похожих на реальные ключи, без "keys"/"embs"
    return EmbBank(mode="perkey", npz_obj=z)


class TripletGraph:
    """
    - Строим ID‑карты по (train ∪ test), чтобы индексы теста были валидны.
    - Рёбра графа строим ТОЛЬКО из (train ∪ valid), чтобы не было утечки теста.
    - Пакуем индексы трёх выборок (train/valid/test) в отдельные тензоры.
    - Вместо nn.Embedding используем предвычисленные pLM вектора в data[*].x
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

        # банки эмбеддингов
        self.pep_bank = load_bank('/home/team/data/embeddings/pep_embs_biobert.npz')
        self.tcr_bank = load_bank('/home/team/data/embeddings/tcr_data.npz')
        self.mhc_bank = load_bank('/home/team/data/embeddings/mhc_data.npz')

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

        # ---------- ВАЖНО: подкладываем pLM эмбеддинги вместо nn.Embedding ----------
        data["pep"].x = self.pep_bank.subset_by_vocab(self.pid, missing="random")  # (|P|, Dp)
        data["mhc"].x = self.mhc_bank.subset_by_vocab(self.mid, missing="random")  # (|M|, Dm)
        data["tcr"].x = self.tcr_bank.subset_by_vocab(self.tid, missing="random")  # (|T|, Dt)

        self.data = data
        return data


class TripletGAT(nn.Module):
    """
    Берёт предвычисленные эмбеддинги из data['pep'|'mhc'|'tcr'].x,
    проецирует их в общее пространство hidden и прогоняет через GATv2 на гетерографе.
    """
    def __init__(self,
                 in_dim_pep: int,
                 in_dim_mhc: int,
                 in_dim_tcr: int,
                 hidden: int = 256,
                 layers: int = 2,
                 dropout: float = 0.2,
                 heads: int = 4):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        self.hidden = hidden
        self.layers_n = layers

        # Проекции входных размерностей узлов к единому hidden
        self.in_proj = nn.ModuleDict({
            "pep": nn.Linear(in_dim_pep, hidden),
            "mhc": nn.Linear(in_dim_mhc, hidden),
            "tcr": nn.Linear(in_dim_tcr, hidden),
        })

        # Стек гетеро‑GAT слоёв
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                HeteroConv({
                    ("pep", "binds", "mhc"): GATv2Conv(
                        (-1, -1), hidden, heads=heads, concat=False, dropout=dropout,
                        add_self_loops=False
                    ),
                    ("mhc", "presents_to", "tcr"): GATv2Conv(
                        (-1, -1), hidden, heads=heads, concat=False, dropout=dropout,
                        add_self_loops=False
                    ),
                }, aggr="mean")
            )

        # Голова на тройку (pep, mhc, tcr)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _initial_h(self, data: HeteroData, device) -> Dict[str, torch.Tensor]:
        # берём предвычисленные признаки, проецируем в hidden, делаем нелинейность+дропаут
        h = {}
        for ntype in ["pep", "mhc", "tcr"]:
            if "x" not in data[ntype]:
                raise ValueError(f"data['{ntype}'].x отсутствует — положи туда предвычисленные эмбеддинги")
            x = data[ntype].x.to(device)
            x = self.in_proj[ntype](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            h[ntype] = x
        return h

    def _node_embs(self, data: HeteroData, device) -> Dict[str, torch.Tensor]:
        h = self._initial_h(data, device)

        edge_index_dict = {
            ("pep", "binds", "mhc"): data["pep", "binds", "mhc"].edge_index,
            ("mhc", "presents_to", "tcr"): data["mhc", "presents_to", "tcr"].edge_index,
        }

        # прогон через стек HeteroConv с residual, ELU, Dropout
        for conv in self.layers:
            out = conv(h, edge_index_dict)
            new_h = {}
            for k in h.keys():
                x = out.get(k, h[k])
                if x.shape[-1] == h[k].shape[-1]:
                    x = x + h[k]  # residual
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                new_h[k] = x
            h = new_h
        return h

    def forward_logits(self, data: HeteroData, device, pack: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self._node_embs(data, device)
        hp = h["pep"][pack["pep"]]
        hm = h["mhc"][pack["mhc"]]
        ht = h["tcr"][pack["tcr"]]
        return self.head(torch.cat([hp, hm, ht], dim=-1)).squeeze(-1)


class Runner:
    def __init__(self, cfg: Config, data: HeteroData):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.data = data

        # переносим рёбра на девайс
        for et in [("pep","binds","mhc"), ("mhc","presents_to","tcr")]:
            data[et].edge_index = data[et].edge_index.to(self.device)
        # пакеты индексов
        for split in ["triplet_train","triplet_valid","triplet_test"]:
            for k,v in data[split].items():
                data[split][k] = v.to(self.device)

        self.model = TripletGAT(in_dim_pep = int(data["pep"].x.size(1)),
                                in_dim_mhc = int(data["mhc"].x.size(1)),
                                in_dim_tcr = int(data["tcr"].x.size(1)),
                                hidden = self.cfg.hidden,
                                layers = self.cfg.layers,
                                dropout = self.cfg.dropout,
                                heads = 4).to(self.device)

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