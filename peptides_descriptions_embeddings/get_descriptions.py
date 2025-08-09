import json
import pandas as pd
import os

from typing import Optional, List, Set
from dataclasses import dataclass

import time
import asyncio
import aiohttp
from tqdm.asyncio import tqdm


@dataclass
class PeptideInfo:
    sequence: str
    reference_titles: str = "no titles"
    assay_names: str = "no assay names"
    disease_names: str = "no disease names"
    parent_source_antigen_names: str = "no parent source antigen names"
    parent_source_antigen_source_org_names: str = "no parent source antigen source org names"


async def generate_summary_async(sequence: str) -> Optional[PeptideInfo]:
    host_address = "https://query-api.iedb.org"

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{host_address}/epitope_search?linear_sequence=eq.{sequence}") as resp_search:

            try:
                output_search = json.loads(await resp_search.text())[0]
            except:
                print(f"Broken api output for sequence {sequence}")
                return PeptideInfo(sequence=sequence)

    result = PeptideInfo(sequence=sequence)

    if output_search["reference_titles"] is not None:
        result.reference_titles = ",".join(output_search["reference_titles"][0].split("|"))
    if output_search["assay_names"] is not None:
        result.assay_names = ",".join(output_search["assay_names"][0].split("|"))
    if output_search["disease_names"] is not None:
        result.disease_names = ",".join(output_search["disease_names"][0].split("|"))
    if output_search["parent_source_antigen_names"] is not None:
        result.parent_source_antigen_names = ",".join(output_search["parent_source_antigen_names"][0].split("|"))
    if output_search["parent_source_antigen_source_org_names"] is not None:
        ",".join(output_search["parent_source_antigen_source_org_names"][0].split("|"))

    return result


key = "unique_peptides"
train_data = pd.read_csv(f"{key}.csv")

async def main(i: int) -> None:
    if i * 100 > len(train_data):
        return
    coroutines = []

    for _, row in train_data[i * 100: (i + 1) * 100].iterrows():
        coroutines.append(generate_summary_async(sequence=row["sequence"]))

    results = []
    for f in tqdm.as_completed(coroutines):
        results.append(await f)

    df_raw = []
    for result in results:
        df_raw.append(
            result.__dict__
        )


    df = pd.DataFrame(df_raw)
    df.to_csv(f"{key}_descriptions.csv", mode="a", header=(not os.path.exists(f"{key}_descriptions.csv")), index=False)


for i in range(len(train_data) // 100 + 1):
    asyncio.run(main(i))
    time.sleep(5)
