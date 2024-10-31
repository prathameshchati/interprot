import csv
import math
from typing import Callable, TextIO

import click
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from interprot.sae_model import SparseAutoencoder
from interprot.utils import get_layer_activations


def compute_scores_matrix(
    sequence_target: list[tuple[str, np.ndarray]],
    sequence2latents: Callable[[str], torch.Tensor],
    sae_dim: int,
) -> np.ndarray:
    """
    Given a list of tuples like [(MVLSEGEWQL, 0001111110), ...] and a function
    that can convert a protein sequence to SAE latents, returns a matrix of
    scores like this:

    +----------------+----------------+----------------+----------------+
    | Sequence       | SAE Dim 1      | SAE Dim 2      | ...            |
    +----------------+----------------+----------------+----------------+
    | MVLSEGE...     | 1.90011590     | 1.27311590     | ...            |
    | PPYTVVY...     | 0.00011020     | 0.00032821     | ...            |
    | ...            | ...            | ...            | ...            |
    +----------------+----------------+----------------+----------------+

    For each sequence i and SAE latent dimension j, `scores[i, j]` is a
    measurement of how strongly the activation of SAE latent dim j correlates
    with the target label for sequence i.

    The scoring calculation is as follows. For each sequence, get its SAE latents,
    a vector of length `sae_dim`.
    1. Group these latent activations into two bins: positive vs. negative.
        - positive: activations whose position has the target label 1
        - negative: activations whose position has the target label 0
    2. Do a 2-sample t-test to see if the mean activation of the positive bin is
       statistically different from the mean activation of the negative bin.

       t = (mean1 - mean0) / sqrt((var1/n1) + (var0/n0))

       Where:
       - mean1, var1, n1 are the mean, variance, and sample size of group 1
       - mean0, var0, n0 are the mean, variance, and sample size of group 0
    """
    scores = np.zeros((len(sequence_target), sae_dim))

    for seq_idx, (sequence, target) in tqdm(
        enumerate(sequence_target),
        total=len(sequence_target),
        desc="Processing sequences",
    ):
        sae_acts = sequence2latents(sequence)
        for dim_idx in range(sae_dim):
            hidden_dim_acts = sae_acts[:, dim_idx]

            # For most SAE latent dims (all but K), the activations are all 0.
            # Skip them and let scores default to 0.
            if torch.all(hidden_dim_acts == 0).item():
                continue

            positive_acts = hidden_dim_acts[target == 1]
            positive_acts_mean = positive_acts.mean().item()
            positive_acts_var = positive_acts.var().item()

            negative_acts = hidden_dim_acts[target == 0]
            negative_acts_mean = negative_acts.mean().item()
            negative_acts_var = negative_acts.var().item()

            score = (positive_acts_mean - negative_acts_mean) / math.sqrt(
                positive_acts_var / len(positive_acts) + negative_acts_var / len(negative_acts)
            )
            scores[seq_idx, dim_idx] = score

    return scores


@click.command
@click.option(
    "--labels-csv",
    type=click.File("r"),
    required=True,
    help="CSV file containing sequence labels",
)
@click.option(
    "--sae-checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to the SAE checkpoint file",
)
@click.option("--plm-dim", type=int, required=True, help="Dimension of the protein language model")
@click.option("--sae-dim", type=int, required=True, help="Dimension of the sparse autoencoder")
@click.option(
    "--plm-layer",
    type=int,
    required=True,
    help="Layer of the protein language model to use",
)
@click.option(
    "--out-path",
    type=click.Path(),
    required=True,
    help="Path to save the output CSV file",
)
@click.option("--max-seqs", type=int, default=100, help="Maximum number of sequences to process")
def labels2latents(
    labels_csv: TextIO,
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
    click.echo(f"Processing {labels_csv.name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")

    sequence_target = []
    reader = csv.DictReader(labels_csv)
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

    scores = compute_scores_matrix(sequence_target, sequence2latents, sae_dim)

    # Get the mean score for each SAE dimension, sort in descending order.
    mean_scores = scores.mean(axis=0)
    sae_dim_scores = [(i, score) for i, score in enumerate(mean_scores)]
    sae_dim_scores.sort(key=lambda x: x[1], reverse=True)

    click.echo(f"Mapped activations for {len(sequence_target)} sequences SAE latents.")
    click.echo(f"Writing results to {out_path}...")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sae_dim", "score"])
        writer.writerows(sae_dim_scores)
