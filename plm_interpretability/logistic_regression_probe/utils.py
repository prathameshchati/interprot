import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from plm_interpretability.logistic_regression_probe.annotations import ResidueAnnotation
from plm_interpretability.logistic_regression_probe.logging import logger
from plm_interpretability.sae_model import SparseAutoencoder
from plm_interpretability.utils import get_layer_activations, parse_swissprot_annotation


def get_sae_acts(
    seq: str,
    tokenizer: AutoTokenizer,
    plm_model: EsmModel,
    sae_model: SparseAutoencoder,
    plm_layer: int,
) -> np.ndarray[float]:
    """
    Returns a (len(seq), sae_dim) array of SAE activations.
    """
    esm_layer_acts = get_layer_activations(
        tokenizer=tokenizer, plm=plm_model, seqs=[seq], layer=plm_layer
    )[0]
    sae_acts = sae_model.get_acts(esm_layer_acts)[1:-1]  # Trim BOS and EOS tokens
    return sae_acts.cpu().numpy()


def get_annotation_entries_for_class(
    swissprot_df: pd.DataFrame,
    annotation: ResidueAnnotation,
    class_name: str,
    max_seqs_per_task: int,
) -> dict[str, list[dict]]:
    """
    Map each sequence to a list of annotations entries like:
    {
        "AAA": [
            {"start": 1, "end": 24, "note": "H-T-H motif"},
            {"start": 100, "end": 120, "note": "Homeobox"},
        ],
        ...
    }
    Downsample to max_seqs_per_task if necessary.
    """
    seq_to_annotation_entries = {}
    seq_lengths = []
    for _, row in swissprot_df[swissprot_df[annotation.name].notna()].iterrows():
        seq = row["Sequence"]
        entries = parse_swissprot_annotation(
            row[annotation.name], header=annotation.swissprot_header
        )
        if class_name != ResidueAnnotation.ALL_CLASSES:
            # The note field is sometimes like "Homeobox", "Homeobox 1", etc.,
            # so use string `in` to check.
            entries = [e for e in entries if class_name in e.get("note", "")]
        if len(entries) > 0 and len(seq) < 2000:
            seq_to_annotation_entries[seq] = entries
            seq_lengths.append(len(seq))

    logger.info(
        f"Found {len(seq_to_annotation_entries)} sequences with class {class_name}."
        f"Mean sequence length: {np.mean(seq_lengths):.2f}."
    )

    if len(seq_to_annotation_entries) > max_seqs_per_task:
        logger.warning(
            f"Since max_seqs_per_task={max_seqs_per_task}, using a random "
            f"sample of {max_seqs_per_task} sequences."
        )
        subset_seqs = random.sample(list(seq_to_annotation_entries.keys()), max_seqs_per_task)
        seq_to_annotation_entries = {
            seq: entries for seq, entries in seq_to_annotation_entries.items() if seq in subset_seqs
        }

    return seq_to_annotation_entries


def make_examples_from_annotation_entries(
    seq_to_annotation_entries: dict[str, list[dict]],
    tokenizer: AutoTokenizer,
    plm_model: EsmModel,
    sae_model: SparseAutoencoder,
    plm_layer: int,
):
    """
    Given a dict like this:
    ```
    {
        "AAA": [
            {"start": 1, "end": 24},
            {"start": 100, "end": 120},
        ],
        ...
    }
    ```
    Create an example for each residue in each sequence where:

    Input: SAE activation at the residue position
    Target: Boolean indicating whether the residue has an annotation with
    of given class, e.g. whether it falls within a motif of class
    "H-T-H motif".

    Returns a list of dicts like:
    ```
    [
        {
            "sae_acts": [0.1, 0.2, 0.3, ...], # A number for each latent
            "target": True,
        },
        {
            "sae_acts": [0.4, 0.5, 0.6, ...],
            "target": False,
        },
        ...
    ]
    ```
    """
    examples = []
    num_positive_examples = 0
    for seq, entries in tqdm(
        seq_to_annotation_entries.items(),
        desc="Running ESM -> SAE inference",
    ):
        positive_positions = set()
        for e in entries:
            for i in range(e["start"] - 1, e["end"]):  # Swissprot is 1-indexed
                positive_positions.add(i)

        sae_acts = get_sae_acts(
            seq=seq,
            tokenizer=tokenizer,
            plm_model=plm_model,
            sae_model=sae_model,
            plm_layer=plm_layer,
        )

        for i, pos_sae_acts in enumerate(sae_acts):
            examples.append(
                {
                    "sae_acts": pos_sae_acts,
                    "target": i in positive_positions,
                }
            )
            if i in positive_positions:
                num_positive_examples += 1

    logger.info(f"Made {len(examples)} examples ({num_positive_examples} positive)")
    return examples
