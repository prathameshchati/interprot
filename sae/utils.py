import os
from typing import Optional

import numpy as np
import polars as pl
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def create_file(dir: str, file_name: str) -> None:
    if not os.path.isdir(dir):
        raise ValueError(f"The specified directory '{dir}' does not exist.")

    file_path = os.path.join(dir, file_name)
    try:
        open(file_path, "x").close()
    except FileExistsError:
        pass


def get_layer_activations(
    tokenizer: PreTrainedTokenizer,
    plm: PreTrainedModel,
    seqs: list[str],
    layer: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Get the activations of a specific layer in a pLM model. Let:

    ```
    N = len(seqs)
    L = max(len(seq) for seq in seqs) + 2 # +2 for BOS and EOS tokens
    D_MODEL = the layer dimension of the pLM, i.e. "Embedding Dim" column here
        https://github.com/facebookresearch/esm/tree/main?tab=readme-ov-file#available-models
    ```

    The output tensor is of shape (N, L, D_MODEL)

    Args:
        tokenizer: The tokenizer to use.
        plm: The pLM model to get the activations from.
        seqs: The sequences to get the activations for.
        layer: The layer to get the activations from.
        device: The device to use.

    Returns:
        The (N, L, D_MODEL) activations of the specified layer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(seqs, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = plm(**inputs, output_hidden_states=True)
    layer_acts = outputs.hidden_states[layer]
    del outputs
    return layer_acts


def train_val_test_split(
    df: pl.DataFrame, train_frac: float = 0.9
) -> tuple[list[str], list[str], list[str]]:
    """
    Split the sequences into training, validation, and test sets. train_frac specifies
    the fraction of examples to use for training; the rest is split evenly between
    validation and test.

    Doing this by samples so it's stochastic.

    Args:
        seqs: The sequences to split.
        train_frac: The fraction of examples to use for training.

    Returns:
        A tuple containing the training, validation, and test sets.
    """
    is_train = pl.Series(
        np.random.choice([True, False], size=len(df), p=[train_frac, 1 - train_frac])
    )
    seqs_train = df.filter(is_train)
    seqs_val_test = df.filter(~is_train)

    is_val = pl.Series(
        np.random.choice([True, False], size=len(seqs_val_test), p=[0.5, 0.5])
    )
    seqs_val = seqs_val_test.filter(is_val)
    seqs_test = seqs_val_test.filter(~is_val)
    return seqs_train, seqs_val, seqs_test
