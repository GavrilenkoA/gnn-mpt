import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv


# =================== I/O ===================
TRIP_TRAIN_PATH = Path('/mnt/nfs_protein/gavrilenko/mpt/raw/training_data.csv')   # Antigen,HLA,CDR3
TRIP_TEST_PATH = Path('/mnt/nfs_protein/gavrilenko/mpt/raw/testing_data.csv')

PM_TRAIN_PATH = Path('/mnt/nfs_protein/gavrilenko/mpt/raw/edges_pm_seq_filt.csv')
PT_TRAIN_PATH = Path('/mnt/nfs_protein/gavrilenko/mpt/raw/edges_pt_seq_filt.csv')


def _read_pairs_pm(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p).dropna(subset=["Antigen","HLA"]).drop_duplicates()
    return df[["Antigen","HLA"]].copy()


def _read_pairs_pt(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p).dropna(subset=["Antigen","CDR3"]).drop_duplicates()
    return df[["Antigen","CDR3"]].copy()


def _read_triplets(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p).dropna(subset=["Antigen","HLA","CDR3"]).drop_duplicates()
    return df[["Antigen", "HLA", "CDR3"]].copy()


pm_train_df = _read_pairs_pm(PM_TRAIN_PATH)
pt_train_df = _read_pairs_pt(PT_TRAIN_PATH)
trip_train_df = _read_triplets(TRIP_TRAIN_PATH)
trip_test_df = _read_triplets(TRIP_TEST_PATH)   # тест остаётся ТОЛЬКО тройки


# =================== utils ===================
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def bin_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc":  float(average_precision_score(y_true, y_score)),
    }


def _gen_neg_pairs(
    pos_pairs: List[Tuple[int,int]],
    left_pool: np.ndarray,
    right_pool: np.ndarray,
    forbid_pos: set,  # набор всех известных позитивов этого типа (на train/valid/test)
    k: int = 1,
    seed: int = 42,
    max_tries: int = 50
) -> List[Tuple[int,int]]:
    rng = np.random.default_rng(seed)
    pos_set = set(pos_pairs)
    neg = []
    tries_cap = max_tries * len(pos_pairs) * k
    tries = 0
    while len(neg) < len(pos_pairs) * k and tries < tries_cap:
        a, b = pos_pairs[rng.integers(low=0, high=len(pos_pairs))]
        if rng.random() < 0.5:
            b2 = int(rng.choice(right_pool)); cand = (a, b2)
        else:
            a2 = int(rng.choice(left_pool));  cand = (a2, b)
        if cand not in pos_set and cand not in forbid_pos:
            neg.append(cand)
        tries += 1
    return neg


def _gen_neg_triplets(
    pos_triples: List[Tuple[int,int,int]],
    pep_pool: np.ndarray,
    mhc_pool: np.ndarray,
    tcr_pool: np.ndarray,
    forbid_pos: set,  # все известные позитивы-тройки
    k: int = 1,
    seed: int = 42,
    max_tries: int = 50
) -> List[Tuple[int,int,int]]:
    rng = np.random.default_rng(seed)
    pos_set = set(pos_triples)
    neg = []
    tries_cap = max_tries * len(pos_triples) * k
    tries = 0
    while len(neg) < len(pos_triples) * k and tries < tries_cap:
        p, m, t = pos_triples[rng.integers(low=0, high=len(pos_triples))]
        r = rng.integers(0, 3)
        if r == 0:
            p2 = int(rng.choice(pep_pool)); cand = (p2, m, t)
        elif r == 1:
            m2 = int(rng.choice(mhc_pool)); cand = (p, m2, t)
        else:
            t2 = int(rng.choice(tcr_pool)); cand = (p, m, t2)
        if cand not in pos_set and cand not in forbid_pos:
            neg.append(cand)
        tries += 1
    return neg



@dataclass
class Config:
    seed: int = 42
    val_size: float = 0.2

    emb_dim: int = 128
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    heads: int = 4

    # фаза 1: пары
    pair_epochs: int = 12
    pair_lr: float = 2e-3
    pair_weight_decay: float = 1e-4
    pm_neg_k: int = 1
    pt_neg_k: int = 1

    # фаза 2: тройки
    triplet_epochs: int = 30
    triplet_lr: float = 2e-3
    triplet_weight_decay: float = 1e-4
    triplet_neg_k: int = 2

    # (опционально) совместная регуляризация парами во 2-й фазе
    multitask_in_stage2: bool = False
    lambda_pm: float = 0.25
    lambda_pt: float = 0.25

    device: str = "cuda:1"
    early_patience: int = 6
    ckpt_pairs: str = "best_pairs.pt"
    ckpt_triplet: str = "best_triplet.pt"


# =================== graph builder ===================
class TripletGraph:
    """
    PM/PT берём только из train (и делим на train/valid).
    Тройки — из training_data (делим на train/valid) и testing_data (test).
    Рёбра формируем из train∪valid (без утечки из теста).
    Для негативов запрещаем совпадение с известными позитивами соответствующего типа
    (включая пары, полученные из тройковых датасетов).
    """
    def __init__(self,
                 pm_train_df: pd.DataFrame,
                 pt_train_df: pd.DataFrame,
                 trip_train_df: pd.DataFrame,
                 trip_test_df: pd.DataFrame,
                 cfg: Config):
        self.pm_train_df = pm_train_df.reset_index(drop=True)
        self.pt_train_df = pt_train_df.reset_index(drop=True)
        self.trip_train_df = trip_train_df.reset_index(drop=True)
        self.trip_test_df = trip_test_df.reset_index(drop=True)
        self.cfg = cfg

        # проверки
        for df, cols in [(self.pm_train_df, ["Antigen","HLA"]),
                         (self.pt_train_df, ["Antigen","CDR3"]),
                         (self.trip_train_df, ["Antigen","HLA","CDR3"]),
                         (self.trip_test_df,  ["Antigen","HLA","CDR3"])]:
            for c in cols:
                if c not in df.columns:
                    raise ValueError(f"Missing column '{c}'")

        self.pid = {}; self.mid = {}; self.tid = {}
        self.data: HeteroData = None

    def build_id_maps(self):
        all_p = pd.Index(pd.unique(pd.concat([
            self.pm_train_df["Antigen"], self.pt_train_df["Antigen"],
            self.trip_train_df["Antigen"], self.trip_test_df["Antigen"]
        ])))
        all_m = pd.Index(pd.unique(pd.concat([
            self.pm_train_df["HLA"],
            self.trip_train_df["HLA"], self.trip_test_df["HLA"]
        ])))
        all_t = pd.Index(pd.unique(pd.concat([
            self.pt_train_df["CDR3"],
            self.trip_train_df["CDR3"], self.trip_test_df["CDR3"]
        ])))
        self.pid = {v:i for i,v in enumerate(all_p)}
        self.mid = {v:i for i,v in enumerate(all_m)}
        self.tid = {v:i for i,v in enumerate(all_t)}

    @staticmethod
    def _edge_index(pairs: list) -> torch.Tensor:
        if len(pairs) == 0:
            return torch.empty(2, 0, dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    def build_graph_and_packs(self) -> HeteroData:
        cfg = self.cfg

        # сплиты для PM/PT/Triplet (PM/PT есть только в train-файлах)
        def split_idx(n, seed):
            idx = np.arange(n)
            tr, va = train_test_split(idx, test_size=cfg.val_size, random_state=seed, shuffle=True)
            return tr, va

        pm_tr_idx, pm_va_idx = split_idx(len(self.pm_train_df), cfg.seed)
        pt_tr_idx, pt_va_idx = split_idx(len(self.pt_train_df), cfg.seed+1)
        trip_tr_idx, trip_va_idx = split_idx(len(self.trip_train_df), cfg.seed+2)

        pm_tr = self.pm_train_df.iloc[pm_tr_idx].reset_index(drop=True)
        pm_va = self.pm_train_df.iloc[pm_va_idx].reset_index(drop=True)

        pt_tr = self.pt_train_df.iloc[pt_tr_idx].reset_index(drop=True)
        pt_va = self.pt_train_df.iloc[pt_va_idx].reset_index(drop=True)

        trip_tr = self.trip_train_df.iloc[trip_tr_idx].reset_index(drop=True)
        trip_va = self.trip_train_df.iloc[trip_va_idx].reset_index(drop=True)
        trip_te = self.trip_test_df.reset_index(drop=True)

        # маппинг в ID
        def map_pm(df):
            return pd.DataFrame({"p":[self.pid[p] for p in df["Antigen"].values],
                                 "m":[self.mid[m] for m in df["HLA"].values]})
        def map_pt(df):
            return pd.DataFrame({"p":[self.pid[p] for p in df["Antigen"].values],
                                 "t":[self.tid[t] for t in df["CDR3"].values]})
        def map_trip(df):
            return pd.DataFrame({"p":[self.pid[p] for p in df["Antigen"].values],
                                 "m":[self.mid[m] for m in df["HLA"].values],
                                 "t":[self.tid[t] for t in df["CDR3"].values]})

        pm_tr_id, pm_va_id = map_pm(pm_tr), map_pm(pm_va)
        pt_tr_id, pt_va_id = map_pt(pt_tr), map_pt(pt_va)
        trip_tr_id, trip_va_id, trip_te_id = map_trip(trip_tr), map_trip(trip_va), map_trip(trip_te)

        # глобальные позитивы для запрета в негатив-сэмплинге
        # пары, встречающиеся в тройковых датасетах, тоже считаем позитивными парами
        pm_from_trip_all = set((int(p),int(m)) for p,m in zip(
            pd.concat([trip_tr_id["p"], trip_va_id["p"], trip_te_id["p"]]),
            pd.concat([trip_tr_id["m"], trip_va_id["m"], trip_te_id["m"]])
        ))
        pt_from_trip_all = set((int(p),int(t)) for p,t in zip(
            pd.concat([trip_tr_id["p"], trip_va_id["p"], trip_te_id["p"]]),
            pd.concat([trip_tr_id["t"], trip_va_id["t"], trip_te_id["t"]])
        ))

        pm_all_pos = set(map(tuple, pd.concat([pm_tr_id, pm_va_id]).values.tolist()))
        pm_all_pos |= pm_from_trip_all

        pt_all_pos = set(map(tuple, pd.concat([pt_tr_id, pt_va_id]).values.tolist()))
        pt_all_pos |= pt_from_trip_all

        trip_all_pos = set(map(tuple, pd.concat([trip_tr_id, trip_va_id, trip_te_id]).values.tolist()))

        # позитивы без дубликатов
        pm_tr_pos = list({(int(p),int(m)) for p,m in zip(pm_tr_id["p"], pm_tr_id["m"])})
        pm_va_pos = list({(int(p),int(m)) for p,m in zip(pm_va_id["p"], pm_va_id["m"])})

        pt_tr_pos = list({(int(p),int(t)) for p,t in zip(pt_tr_id["p"], pt_tr_id["t"])})
        pt_va_pos = list({(int(p),int(t)) for p,t in zip(pt_va_id["p"], pt_va_id["t"])})

        trip_tr_pos = list(zip(trip_tr_id["p"].tolist(), trip_tr_id["m"].tolist(), trip_tr_id["t"].tolist()))
        trip_va_pos = list(zip(trip_va_id["p"].tolist(), trip_va_id["m"].tolist(), trip_va_id["t"].tolist()))
        trip_te_pos = list(zip(trip_te_id["p"].tolist(), trip_te_id["m"].tolist(), trip_te_id["t"].tolist()))

        # негативы
        pep_pool = np.arange(len(self.pid)); mhc_pool = np.arange(len(self.mid)); tcr_pool = np.arange(len(self.tid))
        pm_tr_neg = _gen_neg_pairs(pm_tr_pos, pep_pool, mhc_pool, forbid_pos=pm_all_pos, k=cfg.pm_neg_k, seed=cfg.seed)
        pm_va_neg = _gen_neg_pairs(pm_va_pos, pep_pool, mhc_pool, forbid_pos=pm_all_pos, k=cfg.pm_neg_k, seed=cfg.seed+1)

        pt_tr_neg = _gen_neg_pairs(pt_tr_pos, pep_pool, tcr_pool, forbid_pos=pt_all_pos, k=cfg.pt_neg_k, seed=cfg.seed)
        pt_va_neg = _gen_neg_pairs(pt_va_pos, pep_pool, tcr_pool, forbid_pos=pt_all_pos, k=cfg.pt_neg_k, seed=cfg.seed+1)

        trip_tr_neg = _gen_neg_triplets(trip_tr_pos, pep_pool, mhc_pool, tcr_pool, forbid_pos=trip_all_pos, k=cfg.triplet_neg_k, seed=cfg.seed)
        trip_va_neg = _gen_neg_triplets(trip_va_pos, pep_pool, mhc_pool, tcr_pool, forbid_pos=trip_all_pos, k=cfg.triplet_neg_k, seed=cfg.seed+1)
        trip_te_neg = _gen_neg_triplets(trip_te_pos, pep_pool, mhc_pool, tcr_pool, forbid_pos=trip_all_pos, k=cfg.triplet_neg_k, seed=cfg.seed+2)

        data = HeteroData()
        data["pep"].num_nodes = len(self.pid)
        data["mhc"].num_nodes = len(self.mid)
        data["tcr"].num_nodes = len(self.tid)

        # рёбра: из train∪valid наборов
        pm_edges = list({(int(p), int(m)) for p,m in pd.concat([pm_tr_id, pm_va_id]).values.tolist()})
        pt_edges = list({(int(p), int(t)) for p,t in pd.concat([pt_tr_id, pt_va_id]).values.tolist()})
        mt_edges = list({(int(m), int(t)) for m,t in zip(pd.concat([trip_tr_id["m"], trip_va_id["m"]]),
                                                         pd.concat([trip_tr_id["t"], trip_va_id["t"]]))})

        data["pep","binds","mhc"].edge_index = self._edge_index(pm_edges)
        data["pep","contacts","tcr"].edge_index = self._edge_index(pt_edges)
        data["mhc","presents_to","tcr"].edge_index = self._edge_index(mt_edges)

        # пакеты
        def pack_pairs(pos, neg):
            a_pos = torch.tensor([a for a,_ in pos], dtype=torch.long)
            b_pos = torch.tensor([b for _,b in pos], dtype=torch.long)
            a_neg = torch.tensor([a for a,_ in neg], dtype=torch.long)
            b_neg = torch.tensor([b for _,b in neg], dtype=torch.long)
            a = torch.cat([a_pos, a_neg], 0)
            b = torch.cat([b_pos, b_neg], 0)
            y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).long()
            return {"left": a, "right": b, "y": y}

        def pack_trip(pos, neg):
            p_pos = torch.tensor([p for p,_,_ in pos], dtype=torch.long)
            m_pos = torch.tensor([m for _,m,_ in pos], dtype=torch.long)
            t_pos = torch.tensor([t for _,_,t in pos], dtype=torch.long)
            p_neg = torch.tensor([p for p,_,_ in neg], dtype=torch.long)
            m_neg = torch.tensor([m for _,m,_ in neg], dtype=torch.long)
            t_neg = torch.tensor([t for _,_,t in neg], dtype=torch.long)
            p = torch.cat([p_pos, p_neg]); m = torch.cat([m_pos, m_neg]); t = torch.cat([t_pos, t_neg])
            y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).long()
            return {"pep": p, "mhc": m, "tcr": t, "y": y}

        data["pm_train"] = pack_pairs(pm_tr_pos, pm_tr_neg)
        data["pm_valid"] = pack_pairs(pm_va_pos, pm_va_neg)

        data["pt_train"] = pack_pairs(pt_tr_pos, pt_tr_neg)
        data["pt_valid"] = pack_pairs(pt_va_pos, pt_va_neg)

        data["triplet_train"] = pack_trip(trip_tr_pos, trip_tr_neg)
        data["triplet_valid"] = pack_trip(trip_va_pos, trip_va_neg)
        data["triplet_test"]  = pack_trip(trip_te_pos, trip_te_neg)

        self.data = data
        return data


# =================== model ===================
class TripletGAT(nn.Module):
    def __init__(self, n_pep, n_mhc, n_tcr, emb_dim=128, hidden=256, layers=2, dropout=0.2, heads=4):
        super().__init__()
        self.dropout = dropout

        self.emb = nn.ModuleDict({
            "pep": nn.Embedding(n_pep, emb_dim),
            "mhc": nn.Embedding(n_mhc, emb_dim),
            "tcr": nn.Embedding(n_tcr, emb_dim),
        })

        self.proj_pep = nn.Linear(emb_dim, hidden)
        self.proj_mhc = nn.Linear(emb_dim, hidden)
        self.proj_tcr = nn.Linear(emb_dim, hidden)

        hetero_layers = []
        for _ in range(layers):
            hetero_layers.append(HeteroConv({
                ("pep","binds","mhc"):      GATv2Conv((-1,-1), hidden, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                ("pep","contacts","tcr"):   GATv2Conv((-1,-1), hidden, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                ("mhc","presents_to","tcr"):GATv2Conv((-1,-1), hidden, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
            }, aggr="mean"))
        self.layers = nn.ModuleList(hetero_layers)

        self.pm_head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,1))
        self.pt_head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,1))
        self.triplet_head = nn.Sequential(nn.Linear(3*hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,1))

    def _init_node_states(self, data: HeteroData, device):
        return {
            "pep": self.proj_pep(self.emb["pep"](torch.arange(data["pep"].num_nodes, device=device))),
            "mhc": self.proj_mhc(self.emb["mhc"](torch.arange(data["mhc"].num_nodes, device=device))),
            "tcr": self.proj_tcr(self.emb["tcr"](torch.arange(data["tcr"].num_nodes, device=device))),
        }

    def _propagate(self, data: HeteroData, device):
        h = self._init_node_states(data, device)
        edge_index_dict = {
            ("pep","binds","mhc"):      data["pep","binds","mhc"].edge_index,
            ("pep","contacts","tcr"):   data["pep","contacts","tcr"].edge_index,
            ("mhc","presents_to","tcr"):data["mhc","presents_to","tcr"].edge_index,
        }
        for conv in self.layers:
            out = conv(h, edge_index_dict)
            new_h = {}
            for k in h.keys():
                x = out.get(k, h[k])
                if x.shape[-1] == h[k].shape[-1]:
                    x = x + h[k]
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                new_h[k] = x
            h = new_h
        return h

    def pm_logits(self, data, device, pack):
        h = self._propagate(data, device)
        hp = h["pep"][pack["left"]]; hm = h["mhc"][pack["right"]]
        return self.pm_head(torch.cat([hp, hm], -1)).squeeze(-1)

    def pt_logits(self, data, device, pack):
        h = self._propagate(data, device)
        hp = h["pep"][pack["left"]]; ht = h["tcr"][pack["right"]]
        return self.pt_head(torch.cat([hp, ht], -1)).squeeze(-1)

    def triplet_logits(self, data, device, pack):
        h = self._propagate(data, device)
        hp = h["pep"][pack["pep"]]; hm = h["mhc"][pack["mhc"]]; ht = h["tcr"][pack["tcr"]]
        return self.triplet_head(torch.cat([hp, hm, ht], -1)).squeeze(-1)


# =================== training ===================
class Runner:
    def __init__(self, cfg: Config, data: HeteroData):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.data = data

        for et in [("pep","binds","mhc"), ("pep","contacts","tcr"), ("mhc","presents_to","tcr")]:
            data[et].edge_index = data[et].edge_index.to(self.device)
        for split in ["pm_train","pm_valid","pt_train","pt_valid",
                      "triplet_train","triplet_valid","triplet_test"]:
            for k,v in data[split].items():
                data[split][k] = v.to(self.device)

        self.model = TripletGAT(
            n_pep=data["pep"].num_nodes,
            n_mhc=data["mhc"].num_nodes,
            n_tcr=data["tcr"].num_nodes,
            emb_dim=cfg.emb_dim, hidden=cfg.hidden, layers=cfg.layers,
            dropout=cfg.dropout, heads=cfg.heads
        ).to(self.device)

        self.crit = nn.BCEWithLogitsLoss()
        self.opt_pairs = torch.optim.AdamW(self.model.parameters(), lr=cfg.pair_lr, weight_decay=cfg.pair_weight_decay)
        self.opt_trip  = torch.optim.AdamW(self.model.parameters(), lr=cfg.triplet_lr, weight_decay=cfg.triplet_weight_decay)

    def _pair_step(self, pm_key: str, pt_key: str, train: bool):
        m = self.model; d = self.data
        m.train() if train else m.eval()
        with torch.set_grad_enabled(train):
            logits_pm = m.pm_logits(d, self.device, d[pm_key])
            logits_pt = m.pt_logits(d, self.device, d[pt_key])
            y_pm = d[pm_key]["y"].float(); y_pt = d[pt_key]["y"].float()
            loss_pm = self.crit(logits_pm, y_pm); loss_pt = self.crit(logits_pt, y_pt)
            loss = 0.5*(loss_pm + loss_pt)
            if train:
                self.opt_pairs.zero_grad(); loss.backward(); self.opt_pairs.step()
            with torch.no_grad():
                m_pm = bin_metrics(y_pm.cpu().numpy(), torch.sigmoid(logits_pm).cpu().numpy())
                m_pt = bin_metrics(y_pt.cpu().numpy(), torch.sigmoid(logits_pt).cpu().numpy())
        return float(loss.item()), {"pm": m_pm, "pt": m_pt}

    def _triplet_step(self, key: str, train: bool, multitask: bool=False):
        m = self.model; d = self.data
        m.train() if train else m.eval()
        with torch.set_grad_enabled(train):
            logits_tr = m.triplet_logits(d, self.device, d[key])
            y = d[key]["y"].float()
            loss = self.crit(logits_tr, y)

            if multitask:
                logits_pm = m.pm_logits(d, self.device, d["pm_train" if "train" in key else "pm_valid"])
                logits_pt = m.pt_logits(d, self.device, d["pt_train" if "train" in key else "pt_valid"])
                y_pm = d["pm_train" if "train" in key else "pm_valid"]["y"].float()
                y_pt = d["pt_train" if "train" in key else "pt_valid"]["y"].float()
                loss = loss + self.cfg.lambda_pm*self.crit(logits_pm, y_pm) + self.cfg.lambda_pt*self.crit(logits_pt, y_pt)

            if train:
                self.opt_trip.zero_grad(); loss.backward(); self.opt_trip.step()
            with torch.no_grad():
                mets = bin_metrics(y.cpu().numpy(), torch.sigmoid(logits_tr).cpu().numpy())
        return float(loss.item()), mets

    def fit_eval(self):
        # -------- ФАЗА 1: пары (train/valid) --------
        best_val, best_state = -1.0, None
        patience = 0
        for epoch in range(1, self.cfg.pair_epochs+1):
            tr_loss, tr_m = self._pair_step("pm_train","pt_train", True)
            va_loss, va_m = self._pair_step("pm_valid","pt_valid", False)
            print(json.dumps({"phase":"pairs","epoch":epoch,"train_loss":tr_loss,"train":tr_m,
                              "valid_loss":va_loss,"valid":va_m}))
            val_score = 0.5*(va_m["pm"]["pr_auc"] + va_m["pt"]["pr_auc"])
            if val_score > best_val + 1e-6:
                best_val = val_score
                best_state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                torch.save(best_state, self.cfg.ckpt_pairs)
                patience = 0
            else:
                patience += 1
                if patience >= self.cfg.early_patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(json.dumps({"phase":"pairs","best_val_pr_auc_mean":best_val, "note":"pairs have no test; test is only on triplets"}))

        # -------- ФАЗА 2: тройки (train/valid -> test) --------
        best_val_tr, best_state_tr = -1.0, None
        patience = 0
        for epoch in range(1, self.cfg.triplet_epochs+1):
            tr_loss, tr_m = self._triplet_step("triplet_train", True, multitask=self.cfg.multitask_in_stage2)
            va_loss, va_m = self._triplet_step("triplet_valid", False, multitask=False)
            print(json.dumps({"phase":"triplets","epoch":epoch,"train_loss":tr_loss,"train":tr_m,
                              "valid_loss":va_loss,"valid":va_m}))
            if va_m["pr_auc"] > best_val_tr + 1e-6:
                best_val_tr = va_m["pr_auc"]
                best_state_tr = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                torch.save(best_state_tr, self.cfg.ckpt_triplet)
                patience = 0
            else:
                patience += 1
                if patience >= self.cfg.early_patience:
                    break
        if best_state_tr is not None:
            self.model.load_state_dict(best_state_tr)

        te_loss_triplet, te_m_triplet = self._triplet_step("triplet_test", False, multitask=False)
        out = {
            "pairs_best_val_pr_auc_mean": best_val,
            "triplet_best_val_pr_auc": best_val_tr,
            "triplet_test": te_m_triplet,
            "ckpt_pairs": self.cfg.ckpt_pairs,
            "ckpt_triplet": self.cfg.ckpt_triplet,
        }
        print(json.dumps(out))
        return out


def main():
    cfg = Config()
    set_seed(cfg.seed)

    gb = TripletGraph(pm_train_df, pt_train_df, trip_train_df, trip_test_df, cfg)
    gb.build_id_maps()
    data = gb.build_graph_and_packs()

    runner = Runner(cfg, data)
    res = runner.fit_eval()
    return res


torch.set_float32_matmul_precision("high")
if __name__ == "__main__":
    main()
