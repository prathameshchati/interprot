import heapq
import json
import os
import re
import subprocess
import tempfile
from typing import Any

import click
import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer, EsmModel

from plm_interpretability.sae_model import SparseAutoencoder
from plm_interpretability.utils import get_layer_activations


class TopKHeap:
    def __init__(self, k: int):
        self.k = k
        self.heap: list[tuple[float, int, Any]] = []

    def push(self, item: tuple[float, int, Any]):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def get_items(self) -> list[tuple[float, int, Any]]:
        return sorted(self.heap, reverse=True)


@click.command()
@click.option(
    "--checkpoint-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the SAE checkpoint file",
)
@click.option(
    "--sequences-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the sequences file containing AlphaFoldDB IDs",
)
def make_viz_files(checkpoint_file: str, sequences_file: str):
    """Generate visualization files for SAE latents"""
    click.echo(f"Generating visualization files for {checkpoint_file}")

    pattern = r"plm(\d+).*?l(\d+).*?sae(\d+)"
    matches = re.search(pattern, checkpoint_file)

    if matches:
        plm_dim, plm_layer, sae_dim = map(int, matches.groups())
    else:
        raise ValueError("Checkpoint file must be named in the format plm<n>_l<n>_sae<n>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    plm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval()
    sae_model = SparseAutoencoder(plm_dim, sae_dim).to(device)
    sae_model.load_state_dict(torch.load(checkpoint_file, map_location=device))

    hidden_dim_to_seqs = {dim: TopKHeap(k=10) for dim in range(sae_dim)}
    # Read the sequences file
    df = pl.read_parquet(sequences_file)
    for seq_idx, row in enumerate(df.iter_rows(named=True)):
        seq = row["Sequence"]

        esm_layer_acts = get_layer_activations(
            tokenizer=tokenizer,
            plm=plm_model,
            seqs=[seq],
            layer=plm_layer,
            device=device,
        )[0]
        sae_acts = sae_model.get_acts(esm_layer_acts)[1:-1].cpu().numpy()
        for dim in range(sae_dim):
            sae_dim_acts = sae_acts[:, dim]
            max_act = np.max(sae_dim_acts)
            hidden_dim_to_seqs[dim].push((max_act, seq_idx, sae_dim_acts))

    dim_to_examples = {}
    for dim in range(sae_dim):
        examples = [
            {
                "tokens_acts_list": [round(act, 1) for act in sae_dim_acts],
                "tokens_list": tokenizer(df[seq_idx]["Sequence"].item())["input_ids"][1:-1],
                "alphafold_id": df[seq_idx]["AlphaFoldDB"].item()[:-1],
            }
            for _, seq_idx, sae_dim_acts in hidden_dim_to_seqs[dim].get_items()
        ]
        dim_to_examples[dim] = examples

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        subprocess.run(["git", "clone", "https://github.com/liambai/plm-interp-viz-data.git"])
        output_dir = os.path.join(temp_dir, "plm-interp-viz-data")
        os.chdir(output_dir)

        for dim in range(sae_dim):
            with open(f"{dim}.txt", "w") as f:
                json.dump(dim_to_examples[dim], f)

        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", f"Add viz files for {checkpoint_file}"])
        subprocess.run(["git", "push"])


if __name__ == "__main__":
    make_viz_files()
