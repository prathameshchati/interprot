import csv
from typing import Callable

import click
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from plm_interpretability.sae_model import SparseAutoencoder
from plm_interpretability.utils import get_layer_activations


def compute_z_scores(
    sequence_target: list[tuple[str, np.ndarray]],
    sequence2latents: Callable[[str], torch.Tensor],
    sae_dim: int,
) -> np.ndarray:
    """
    Given a list of tuples like [(MVLSEGEWQL, 0001111110), ...] and a function
    that can convert a protein sequence to SAE latents, returns a matrix of
    z-scores like this:

    +----------------+----------------+----------------+----------------+
    | Sequence       | SAE Dim 1      | SAE Dim 2      | ...            |
    +----------------+----------------+----------------+----------------+
    | MVLSEGE...     | 1.90011590     | 1.27311590     | ...            |
    | PPYTVVY...     | 0.00011020     | 0.00032821     | ...            |
    | ...            | ...            | ...            | ...            |
    +----------------+----------------+----------------+----------------+

    For each sequence i and SAE latent dimension j, z_scores[i, j] is a
    measurement of how strongly the activation of SAE latent dim j correlates
    with the target label for sequence i.
    """
    z_scores = np.zeros((len(sequence_target), sae_dim))

    for seq_idx, (sequence, target) in tqdm(
        enumerate(sequence_target),
        total=len(sequence_target),
        desc="Processing sequences",
    ):
        sae_acts = sequence2latents(sequence)
        for dim_idx in range(sae_dim):
            hidden_dim_acts = sae_acts[:, dim_idx]
            if torch.all(hidden_dim_acts == 0).item():
                continue

            acts_std = hidden_dim_acts.std().item()

            class_1_acts = hidden_dim_acts[target == 1]
            class_1_acts_mean = class_1_acts.mean().item()

            class_0_acts = hidden_dim_acts[target == 0]
            class_0_acts_mean = class_0_acts.mean().item()

            z_score = (class_1_acts_mean - class_0_acts_mean) / acts_std
            z_scores[seq_idx, dim_idx] = z_score
    return z_scores


@click.command
@click.option(
    "--labels-csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the CSV file containing sequence labels",
)
@click.option(
    "--sae-checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to the SAE checkpoint file",
)
@click.option(
    "--plm-dim", type=int, required=True, help="Dimension of the protein language model"
)
@click.option(
    "--sae-dim", type=int, required=True, help="Dimension of the sparse autoencoder"
)
@click.option(
    "--plm-layer",
    type=int,
    required=True,
    help="Layer of the protein language model to use",
)
@click.option(
    "--out-path", type=str, required=True, help="Path to save the output CSV file"
)
@click.option(
    "--max-seqs", type=int, default=100, help="Maximum number of sequences to process"
)
def labels2latents(
    labels_csv: str,
    sae_checkpoint: str,
    plm_dim: int,
    plm_layer: int,
    sae_dim: int,
    out_path: str,
    max_seqs: int,
):
    """
    Takes in a labels CSV file like this

    +----------------+----------------+
    | sequence       | target         |
    +----------------+----------------+
    | MVLSEGEWQL...  | 0001111110...  |
    +----------------+----------------+

    find SAE latents that tend to activate at positions with the 1 label.
    """
    click.echo(f"Processing {labels_csv}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequence_target = []
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_seqs:
                break
            sequence = row["sequence"]
            target = np.array([int(x) for x in row["target"]])
            sequence_target.append((sequence, target))

    sae_model = SparseAutoencoder(plm_dim, sae_dim).to(device)
    sae_model.load_state_dict(torch.load(sae_checkpoint, map_location=device))
    sae_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    plm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    def sequence2latents(sequence: str) -> torch.Tensor:
        """
        Get the SAE latents for a given sequence.
        """
        esm_acts = get_layer_activations(
            tokenizer=tokenizer,
            plm=plm,
            seqs=[sequence],
            layer=plm_layer,
            device=device,
        )[0]
        sae_acts = sae_model.get_acts(esm_acts)
        sae_acts = sae_acts[1:-1]  # Trim BoS & EoS tokens
        return sae_acts

    z_scores = compute_z_scores(sequence_target, sequence2latents, sae_dim)

    # Get the mean z-score for each SAE dimension, sort in descending order.
    mean_z_scores = z_scores.mean(axis=0)
    sae_dim_scores = [(i, score) for i, score in enumerate(mean_z_scores)]
    sae_dim_scores.sort(key=lambda x: x[1], reverse=True)

    click.echo(f"Mapped activations for {len(sequence_target)} sequences SAE latents.")
    click.echo(f"Writing results to {out_path}...")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sae_dim", "z_score"])
        writer.writerows(sae_dim_scores)
