from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def embed_text(text: str, model: AutoModel) -> torch.Tensor:
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    return last_hidden_states

def main(key: str) -> None:
    descriptions_path = f"{key}.csv"
    seq_only_path = f"{key}_sequences.csv"
    embed_path = f"{key}_embeddings.npy"

    df = pd.read_csv(descriptions_path)

    seqs = df["sequence"]
    seqs.to_csv(seq_only_path, index=False)

    descs = df.apply(
        lambda row: " ".join(f"{col}: {row[col]}; " for col in df.columns[1:]),
        axis=1
    ).tolist()

    result = []
    for desc in tqdm(descs):
        embed = embed_text(desc, model)
        result.append(embed.cpu().detach().numpy().mean(axis=0).mean(axis=0))
    np.save(embed_path, np.array(result))


if __name__ == "__main__":
    main(key="unique_peptides")
