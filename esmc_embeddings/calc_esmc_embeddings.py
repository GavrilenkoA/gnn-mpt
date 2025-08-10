from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import transformers

from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
login()  # This requires authentication through HuggingFace token

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


device = "cuda" if torch.cuda.is_available() else "cpu"
client = ESMC.from_pretrained("esmc_600m").to(device)


def make_embedding(sequence):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return logits_output.embeddings

#### MHC ####

mhc_raw_data = pd.read_csv("pseudoseqs.csv")

mhc_alleles = mhc_raw_data["mhc"].tolist()
mhc_raw_data_wo_alleles = mhc_raw_data.drop(columns=["mhc"])
mhc_raw_data_wo_alleles_columns = mhc_raw_data_wo_alleles.columns

mhc_sequences = mhc_raw_data_wo_alleles.apply(
    lambda row: "".join(
        "" if row[col] == 0 else col.split("_")[1]
        for col in mhc_raw_data_wo_alleles_columns
    ),
    axis=1
).tolist()

mhc_embeddings = []
for sequence in tqdm(mhc_sequences):
    mhc_embeddings.append(
        make_embedding(sequence)[0].cpu().detach().numpy().mean(axis=0)
    )

np.savez(
    "mhc_esmc_600m.npz",
    **dict(list(zip(mhc_alleles, mhc_embeddings)))
)


#### Peptides ####
pep_data = pd.read_csv("nodes_peptides.csv")
pep_seqs = pep_data["sequence"].tolist()

pep_embeddings = []
for sequence in tqdm(pep_seqs):
    pep_embeddings.append(
        make_embedding(sequence)[0].cpu().detach().numpy().mean(axis=0)
    )

np.savez(
    "peptides_esmc_600m.npz",
    **dict(list(zip(pep_seqs, pep_embeddings)))
)


#### TCR ####
tcr_data = pd.read_csv("nodes_tcr.csv")
tcr_seqs = tcr_data["sequence"].tolist()

tcr_embeddings = []
for sequence in tqdm(tcr_seqs):
    tcr_embeddings.append(
        make_embedding(sequence)[0].cpu().detach().numpy().mean(axis=0)
    )

np.savez(
    "tcr_esmc_600m.npz",
    **dict(list(zip(tcr_seqs, tcr_embeddings)))
)
