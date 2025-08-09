# pmt_meta_learn.py
# -*- coding: utf-8 -*-
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


# ------------------------- I/O -------------------------
TRAIN_PATH = Path('/mnt/nfs_protein/gavrilenko/mpt/raw/training_data.csv')
TEST_PATH  = Path('/mnt/nfs_protein/gavrilenko/mpt/raw/testing_data.csv')

train_df = pd.read_csv(TRAIN_PATH).drop_duplicates()
test_df  = pd.read_csv(TEST_PATH).drop_duplicates()

# убираем дубликаты (P,M,T) между train/test
merged_df = test_df.merge(train_df, on=['Antigen', 'HLA', 'CDR3'], how='left', indicator=True)
test_df = test_df[~test_df.index.isin(merged_df[merged_df['_merge'] == 'both'].index)]


# ------------------------- утилиты -------------------------
def generate_triplet_negatives(
    df: pd.DataFrame,
    pep_col: str = "Antigen",
    mhc_col: str = "HLA",
    tcr_col: str = "CDR3",
    k: int = 1,
    seed: int = 42,
    max_tries_per_sample: int = 50,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_mhc = df[mhc_col].unique()
    all_tcr = df[tcr_col].unique()
    seen_pairs_by_pep = {
        pep: set(zip(g[mhc_col].tolist(), g[tcr_col].tolist()))
        for pep, g in df.groupby(pep_col, sort=False)
    }
    neg_rows = []
    for pep, g in df.groupby(pep_col, sort=False):
        n_pos = len(g); need = n_pos * k
        seen_pairs = set(seen_pairs_by_pep[pep]); chosen_pairs = set()
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


def generate_pair_negatives(df_pairs: pd.DataFrame, left: str, right: str,
                            k_ratio: float = 1.0, seed: int = 123) -> pd.DataFrame:
    """Генерим негативы для (left,right) рандомной перестановкой right."""
    rng = np.random.default_rng(seed)
    all_right = df_pairs[right].unique()
    pos = df_pairs.drop_duplicates([left, right]).copy()
    pos["label"] = 1
    n_neg = int(len(pos) * k_ratio)
    neg_left = rng.choice(pos[left].values, size=n_neg, replace=True)
    neg_right = rng.choice(all_right, size=n_neg, replace=True)
    pos_set = set(zip(pos[left], pos[right]))
    keep = []
    for l, r in zip(neg_left, neg_right):
        if (l, r) not in pos_set:
            keep.append({left: l, right: r})
    neg = pd.DataFrame(keep, columns=[left, right])
    neg["label"] = 0
    return pd.concat([pos, neg], ignore_index=True)


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def bin_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc":  float(average_precision_score(y_true, y_score)),
    }


# размечаем тройки
train_df_labeled = generate_triplet_negatives(train_df, k=3, seed=42)
test_df_labeled  = generate_triplet_negatives(test_df,  k=1, seed=123)


# ------------------------- конфиг -------------------------
@dataclass
class Config:
    seed: int = 42
    val_size: float = 0.2

    emb_dim: int = 128      # размер случайных nn.Embedding
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.2
    heads: int = 4

    epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda:0"

    # мультизадачный лосс и гиперы InfoNCE
    lambda_pm: float = 1.0
    lambda_pt: float = 1.0
    lambda_pmt: float = 1.0
    tau: float = 0.1
    pm_neg_ratio: float = 1.0
    pt_neg_ratio: float = 1.0
    pt_num_m_sample: int = 64

    early_patience: int = 6
    ckpt_path: str = "best_meta_triplet.pt"


# ------------------------- graph + пакеты -------------------------
class TripletGraphMeta:
    """
    Гетерограф с рёбрами:
      P–M (pep,binds,mhc), M–T (mhc,presents_to,tcr), P–T (pep,contacts,tcr)
    + обратные рёбра для обновления pep.
    Пакеты:
      triplet_{train,valid,test}: (p,m,t,y)
      pm_train: (p,m,y)
      pt_train: (p,t,y)
    """
    def __init__(self, df_train_labeled: pd.DataFrame, df_test_labeled: pd.DataFrame, cfg: Config):
        self.df_tr_all = df_train_labeled.reset_index(drop=True)
        self.df_te_all = df_test_labeled.reset_index(drop=True)
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

    def _edge_index(self, pairs: List[Tuple[int,int]]) -> torch.Tensor:
        if len(pairs) == 0:
            return torch.empty(2, 0, dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    def build_graph_and_packs(self) -> HeteroData:
        tr_idx, va_idx = train_test_split(
            np.arange(len(self.df_tr_all)),
            test_size=self.cfg.val_size, random_state=self.cfg.seed,
            stratify=self.df_tr_all["label"].values
        )
        df_tr = self.df_tr_all.iloc[tr_idx].reset_index(drop=True)
        df_va = self.df_tr_all.iloc[va_idx].reset_index(drop=True)
        df_te = self.df_te_all.reset_index(drop=True)

        # рёбра из train∪valid
        df_edges = pd.concat([df_tr, df_va], ignore_index=True)

        data = HeteroData()
        data["pep"].num_nodes = len(self.pid)
        data["mhc"].num_nodes = len(self.mid)
        data["tcr"].num_nodes = len(self.tid)

        # прямые рёбра
        pm_edges = [(self.pid[p], self.mid[m]) for p,m in zip(df_edges["Antigen"], df_edges["HLA"])]
        data["pep","binds","mhc"].edge_index = self._edge_index(pm_edges)

        mt_edges = [(self.mid[m], self.tid[t]) for m,t in zip(df_edges["HLA"], df_edges["CDR3"])]
        data["mhc","presents_to","tcr"].edge_index = self._edge_index(mt_edges)

        pt_edges = [(self.pid[p], self.tid[t]) for p,t in zip(df_edges["Antigen"], df_edges["CDR3"])]
        data["pep","contacts","tcr"].edge_index = self._edge_index(pt_edges)

        # обратные рёбра
        data["mhc","bound_by","pep"].edge_index = data["pep","binds","mhc"].edge_index.flip(0)
        data["tcr","contacted_by","pep"].edge_index = data["pep","contacts","tcr"].edge_index.flip(0)

        # упаковщики индексов
        def pack_triplets(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
            return {
                "pep": torch.tensor([self.pid[p] for p in df["Antigen"].values], dtype=torch.long),
                "mhc": torch.tensor([self.mid[m] for m in df["HLA"].values],     dtype=torch.long),
                "tcr": torch.tensor([self.tid[t] for t in df["CDR3"].values],    dtype=torch.long),
                "y":   torch.tensor(df["label"].astype("int64").values),
            }

        data["triplet_train"] = pack_triplets(df_tr)
        data["triplet_valid"] = pack_triplets(df_va)
        data["triplet_test"]  = pack_triplets(df_te)

        # наборы пар (из train∪valid) для multitask
        pm_pairs = df_edges[["Antigen","HLA"]].drop_duplicates()
        pt_pairs = df_edges[["Antigen","CDR3"]].drop_duplicates()

        pm_train = generate_pair_negatives(pm_pairs, left="Antigen", right="HLA",
                                           k_ratio=self.cfg.pm_neg_ratio, seed=self.cfg.seed)
        pt_train = generate_pair_negatives(pt_pairs, left="Antigen", right="CDR3",
                                           k_ratio=self.cfg.pt_neg_ratio, seed=self.cfg.seed)

        def pack_pm(dfpm: pd.DataFrame) -> Dict[str, torch.Tensor]:
            return {
                "pep": torch.tensor([self.pid[p] for p in dfpm["Antigen"].values], dtype=torch.long),
                "mhc": torch.tensor([self.mid[m] for m in dfpm["HLA"].values],     dtype=torch.long),
                "y":   torch.tensor(dfpm["label"].astype("int64").values),
            }

        def pack_pt(dfpt: pd.DataFrame) -> Dict[str, torch.Tensor]:
            return {
                "pep": torch.tensor([self.pid[p] for p in dfpt["Antigen"].values], dtype=torch.long),
                "tcr": torch.tensor([self.tid[t] for t in dfpt["CDR3"].values],    dtype=torch.long),
                "y":   torch.tensor(dfpt["label"].astype("int64").values),
            }

        data["pm_train"] = pack_pm(pm_train)
        data["pt_train"] = pack_pt(pt_train)
        self.data = data
        return data


# ------------------------- модель -------------------------
class TripletGATMeta(nn.Module):
    """
    nn.Embedding -> Hetero GAT -> задачи:
      PM:  v_pm = f_pm([hp,hm]);  P_pm = σ(w_pm(v_pm))
      PMT: v_mt = f_mt([hm,ht]);  логиты через DMF(v_pm ⊙ v_mt)
      PT:  усреднение P_pm по M (аппроксимация)
    """
    def __init__(self, n_pep, n_mhc, n_tcr, emb_dim=128, hidden=256, layers=1, dropout=0.2, heads=4):
        super().__init__()
        self.dropout = dropout
        self.heads = heads

        # обучаемые случайные эмбеддинги
        self.emb = nn.ModuleDict({
            "pep": nn.Embedding(n_pep, emb_dim),
            "mhc": nn.Embedding(n_mhc, emb_dim),
            "tcr": nn.Embedding(n_tcr, emb_dim),
        })

        # один HeteroConv (можно нарастить layers>1 при желании)
        self.gnn = HeteroConv({
            ("pep","binds","mhc"): GATv2Conv((-1,-1), hidden, heads=heads, concat=False,
                                             dropout=dropout, add_self_loops=False),
            ("mhc","presents_to","tcr"): GATv2Conv((-1,-1), hidden, heads=heads, concat=False,
                                                   dropout=dropout, add_self_loops=False),
            ("pep","contacts","tcr"): GATv2Conv((-1,-1), hidden, heads=heads, concat=False,
                                                dropout=dropout, add_self_loops=False),
            # reverse
            ("mhc","bound_by","pep"): GATv2Conv((-1,-1), hidden, heads=heads, concat=False,
                                                dropout=dropout, add_self_loops=False),
            ("tcr","contacted_by","pep"): GATv2Conv((-1,-1), hidden, heads=heads, concat=False,
                                                    dropout=dropout, add_self_loops=False),
        }, aggr="mean")

        # проекции в hidden для согласования размерностей в головах
        self.proj_pep = nn.Linear(emb_dim, hidden)
        self.proj_mhc = nn.Identity()   # после GAT уже hidden
        self.proj_tcr = nn.Identity()

        # головы
        self.f_pm = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )  # v_pm
        self.f_mt = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )  # v_mt
        self.f_dmf = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )  # логит PMT
        self.w_pm = nn.Linear(hidden, 1)  # σ(w_pm(v_pm))

        for mod in [self.proj_pep, self.f_pm, self.f_mt, self.f_dmf, self.w_pm]:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _node_embs(self, data: HeteroData, device) -> Dict[str, torch.Tensor]:
        # начальные эмбеддинги
        h = {
            "pep": self.emb["pep"](torch.arange(data["pep"].num_nodes, device=device)),
            "mhc": self.emb["mhc"](torch.arange(data["mhc"].num_nodes, device=device)),
            "tcr": self.emb["tcr"](torch.arange(data["tcr"].num_nodes, device=device)),
        }
        # message passing
        edge_index_dict = {
            ("pep","binds","mhc"): data["pep","binds","mhc"].edge_index,
            ("mhc","presents_to","tcr"): data["mhc","presents_to","tcr"].edge_index,
            ("pep","contacts","tcr"): data["pep","contacts","tcr"].edge_index,
            ("mhc","bound_by","pep"): data["mhc","bound_by","pep"].edge_index,
            ("tcr","contacted_by","pep"): data["tcr","contacted_by","pep"].edge_index,
        }
        out = self.gnn(h, edge_index_dict)
        for k in h:
            if k in out:
                x = out[k]
                # residual если совпадает размер
                if x.shape[-1] == h[k].shape[-1]:
                    x = x + h[k]
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                h[k] = x
        return h

    def _project(self, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hp = h["pep"]
        if hp.size(-1) == self.proj_pep.in_features:  # обычно 128
            hp = self.proj_pep(hp)
        hm = h["mhc"]
        ht = h["tcr"]
        return {"pep": hp, "mhc": hm, "tcr": ht}


    def pm_logits(self, hp, hm):
        v_pm = self.f_pm(torch.cat([hp, hm], dim=-1))     # [B, hidden]
        logit = self.w_pm(v_pm).squeeze(-1)               # [B]
        return logit, v_pm

    def mt_repr(self, hm, ht):
        v_mt = self.f_mt(torch.cat([hm, ht], dim=-1))     # [B, hidden]
        return v_mt

    def pmt_logits_from_repr(self, v_pm, v_mt):
        inter = v_pm * v_mt                                # DMF: ⊙
        logit = self.f_dmf(inter).squeeze(-1)             # [B]
        return logit


# ------------------------- обучение -------------------------
class RunnerMeta:
    def __init__(self, cfg: Config, data: HeteroData):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.data = data

        # перенос рёбер на девайс
        for et in [
            ("pep","binds","mhc"),
            ("mhc","presents_to","tcr"),
            ("pep","contacts","tcr"),
            ("mhc","bound_by","pep"),
            ("tcr","contacted_by","pep"),
        ]:
            data[et].edge_index = data[et].edge_index.to(self.device)

        # перенос пакетов
        for split in ["triplet_train", "triplet_valid", "triplet_test", "pm_train", "pt_train"]:
            for k, v in data[split].items():
                data[split][k] = v.to(self.device)

        self.model = TripletGATMeta(
            n_pep=data["pep"].num_nodes,
            n_mhc=data["mhc"].num_nodes,
            n_tcr=data["tcr"].num_nodes,
            emb_dim=cfg.emb_dim, hidden=cfg.hidden, layers=cfg.layers,
            dropout=cfg.dropout, heads=cfg.heads,
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.bce = nn.BCEWithLogitsLoss()

        self.all_m_idx = torch.arange(self.data["mhc"].num_nodes, device=self.device)

    @torch.no_grad()
    def _scores_pmt(self, pack: Dict[str, torch.Tensor]) -> torch.Tensor:
        m = self.model; d = self.data
        h = m._node_embs(d, self.device)
        h = m._project(h)  # <-- приводим все к hidden
        hp = h["pep"][pack["pep"]]
        hm = h["mhc"][pack["mhc"]]
        ht = h["tcr"][pack["tcr"]]
        _, v_pm = m.pm_logits(hp, hm)
        v_mt   = m.mt_repr(hm, ht)
        pmt_logit = m.pmt_logits_from_repr(v_pm, v_mt)
        return torch.sigmoid(pmt_logit)

    def _train_step(self):
        m = self.model; d = self.data
        m.train()
        self.opt.zero_grad()

        # единоразово считаем эмбеддинги узлов и приводим в hidden
        h = m._node_embs(d, self.device)
        h = m._project(h)
        h_pep, h_mhc, h_tcr = h["pep"], h["mhc"], h["tcr"]

        # ----- P–M (BCE) -----
        pm = d["pm_train"]
        hp_pm = h_pep[pm["pep"]]
        hm_pm = h_mhc[pm["mhc"]]
        pm_logit, _ = m.pm_logits(hp_pm, hm_pm)
        loss_pm = self.bce(pm_logit, pm["y"].float())

        # ----- P–T (BCE c усреднением по подмножеству M) -----
        pt = d["pt_train"]
        hp_pt = h_pep[pt["pep"]]
        ht_pt = h_tcr[pt["tcr"]]
        M = self.cfg.pt_num_m_sample
        rand_m = self.all_m_idx[torch.randint(0, len(self.all_m_idx), (len(hp_pt), M), device=self.device)]
        hm_s = h_mhc[rand_m]                                # [B, M, H]
        hp_exp = hp_pt.unsqueeze(1).expand_as(hm_s)         # [B, M, H]
        pm_logits_s, _ = m.pm_logits(hp_exp.reshape(-1, hp_pt.size(-1)),
                                     hm_s.reshape(-1, hm_s.size(-1)))
        p_pm = torch.sigmoid(pm_logits_s).reshape(-1, M)
        p_pt = p_pm.mean(dim=1)
        loss_pt = F.binary_cross_entropy(p_pt, pt["y"].float())

        # ----- P–M–T (InfoNCE, DMF) -----
        tr = d["triplet_train"]
        hp_tr = h_pep[tr["pep"]]
        hm_tr = h_mhc[tr["mhc"]]
        ht_tr = h_tcr[tr["tcr"]]
        _, v_pm_tr = m.pm_logits(hp_tr, hm_tr)
        v_mt_tr = m.mt_repr(hm_tr, ht_tr)
        pos_logit = m.pmt_logits_from_repr(v_pm_tr, v_mt_tr)        # [B]

        # within-batch negatives: permute T -> v_mt_neg
        with torch.no_grad():
            idx_perm = torch.randperm(len(v_mt_tr), device=self.device)
        v_mt_neg = v_mt_tr[idx_perm]
        neg_logit = m.pmt_logits_from_repr(v_pm_tr, v_mt_neg)

        tau = self.cfg.tau
        l_pos = torch.exp(pos_logit / tau)
        l_neg = torch.exp(neg_logit / tau)
        loss_pmt = -torch.log(l_pos / (l_pos + l_neg)).mean()

        # ----- общий лосс -----
        loss = self.cfg.lambda_pm * loss_pm + self.cfg.lambda_pt * loss_pt + self.cfg.lambda_pmt * loss_pmt
        loss.backward(); self.opt.step()

        with torch.no_grad():
            scores = torch.sigmoid(pos_logit).detach().cpu().numpy()
            y_np = tr["y"].detach().cpu().numpy()
            mets_tr = bin_metrics(y_np, scores)

        return {
            "loss_total": float(loss.item()),
            "loss_pm": float(loss_pm.item()),
            "loss_pt": float(loss_pt.item()),
            "loss_pmt": float(loss_pmt.item()),
            "train_pr_auc_triplet": float(mets_tr["pr_auc"]),
        }

    @torch.no_grad()
    def _eval_split_triplet(self, split_key: str):
        self.model.eval()
        scores = self._scores_pmt(self.data[split_key]).cpu().numpy()
        y_np = self.data[split_key]["y"].cpu().numpy()
        return bin_metrics(y_np, scores)

    def fit_eval(self):
        best_val_pr = -1.0
        best_state = None
        patience = 0

        for epoch in range(1, self.cfg.epochs + 1):
            log_tr = self._train_step()
            va_m = self._eval_split_triplet("triplet_valid")

            log = {"epoch": epoch, **log_tr, "valid": va_m}
            print(json.dumps(log, ensure_ascii=False))

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

        if best_state is not None:
            self.model.load_state_dict(best_state)

        te_m = self._eval_split_triplet("triplet_test")
        print(json.dumps({"best_val_pr_auc": best_val_pr, "test": te_m, "ckpt": self.cfg.ckpt_path},
                         ensure_ascii=False))
        return {"best_val_pr_auc": best_val_pr, "test": te_m, "ckpt": self.cfg.ckpt_path}


# ------------------------- main -------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)

    gb = TripletGraphMeta(train_df_labeled, test_df_labeled, cfg)
    gb.build_id_maps()
    data = gb.build_graph_and_packs()

    runner = RunnerMeta(cfg, data)
    res = runner.fit_eval()
    return res


torch.set_float32_matmul_precision("high")
if __name__ == "__main__":
    main()
