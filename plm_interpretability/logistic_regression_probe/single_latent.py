import functools
import gc
import os
import tempfile
import warnings
from multiprocessing import Pool

import click
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from plm_interpretability.logistic_regression_probe.annotations import RESIDUE_ANNOTATIONS
from plm_interpretability.logistic_regression_probe.logging import logger
from plm_interpretability.logistic_regression_probe.utils import (
    get_annotation_entries_for_class,
    make_examples_from_annotation_entries,
)
from plm_interpretability.sae_model import SparseAutoencoder


def run_logistic_regression_on_latent(
    dim: int,
    X_train_filename: str,
    y_train_filename: str,
    X_test_filename: str,
    y_test_filename: str,
    shape_train: tuple[int, int],
    shape_test: tuple[int, int],
) -> tuple[int, float, float, float]:
    # Load data from memory-mapped files
    X_train = np.memmap(X_train_filename, dtype="float32", mode="r", shape=shape_train)
    y_train = np.memmap(y_train_filename, dtype="bool", mode="r", shape=(shape_train[0],))
    X_test = np.memmap(X_test_filename, dtype="float32", mode="r", shape=shape_test)
    y_test = np.memmap(y_test_filename, dtype="bool", mode="r", shape=(shape_test[0],))

    X_train_dim = X_train[:, dim].reshape(-1, 1)
    X_test_dim = X_test[:, dim].reshape(-1, 1)

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train_dim, y_train)
    y_pred = model.predict(X_test_dim)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    del X_train, y_train, X_test, y_test, X_train_dim, X_test_dim, model, y_pred
    return (dim, precision, recall, f1)


@click.command()
@click.option(
    "--sae-checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to the SAE checkpoint file",
)
@click.option("--sae-dim", type=int, required=True, help="Dimension of the sparse autoencoder")
@click.option("--plm-dim", type=int, required=True, help="Dimension of the protein language model")
@click.option(
    "--plm-layer",
    type=int,
    required=True,
    help="Layer of the protein language model to use",
)
@click.option(
    "--swissprot-tsv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the SwissProt TSV file",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Path to the output directory",
)
@click.option(
    "--annotation-names",
    type=click.STRING,
    multiple=True,
    help="List of annotation names to process. If not provided, all annotations will be processed.",
)
@click.option(
    "--max-seqs-per-task",
    type=int,
    default=1000,
    help="Maximum number of sequences to use for a given logistic regression task",
)
def single_latent(
    sae_checkpoint: str,
    sae_dim: int,
    plm_dim: int,
    plm_layer: int,
    swissprot_tsv: str,
    output_dir: str,
    annotation_names: list[str],
    max_seqs_per_task: int,
):
    """
    Run 1D logistic regression probing for each latent dimension for SAE evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    # Load pLM and SAE
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    plm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval()
    sae_model = SparseAutoencoder(plm_dim, sae_dim).to(device)
    sae_model.load_state_dict(torch.load(sae_checkpoint, map_location=device))

    df = pd.read_csv(swissprot_tsv, sep="\t")

    for annotation in RESIDUE_ANNOTATIONS:
        if annotation_names and annotation.name not in annotation_names:
            continue

        logger.info(f"Processing annotation: {annotation.name}")
        os.makedirs(os.path.join(output_dir, annotation.name), exist_ok=True)

        for class_name in annotation.class_names:
            output_path = os.path.join(output_dir, annotation.name, f"{class_name}.csv")
            if os.path.exists(output_path):
                logger.warning(f"Skipping {output_path} because it already exists")
                continue

            seq_to_annotation_entries = get_annotation_entries_for_class(
                df, annotation, class_name, max_seqs_per_task
            )
            examples = make_examples_from_annotation_entries(
                seq_to_annotation_entries=seq_to_annotation_entries,
                tokenizer=tokenizer,
                plm_model=plm_model,
                sae_model=sae_model,
                plm_layer=plm_layer,
            )
            # Run logistic regression for each dimension where the input is a number
            # – the SAE activation of a fixed dimension at a fixed position – and
            # the target is the binary target.
            train_examples, test_examples = train_test_split(
                examples,
                test_size=0.1,
                random_state=42,
                stratify=[e["target"] for e in examples],
            )
            with warnings.catch_warnings():
                # LogisticRegression throws warnings when it can't converge.
                # This is expected for most dimensions.
                warnings.simplefilter("ignore")

                X_train = np.array([e["sae_acts"] for e in train_examples], dtype="float32")
                y_train = np.array([e["target"] for e in train_examples], dtype="bool")
                X_test = np.array([e["sae_acts"] for e in test_examples], dtype="float32")
                y_test = np.array([e["target"] for e in test_examples], dtype="bool")

                # Create memory-mapped files
                with tempfile.TemporaryDirectory() as temp_dir:
                    X_train_filename = os.path.join(temp_dir, "X_train.dat")
                    y_train_filename = os.path.join(temp_dir, "y_train.dat")
                    X_test_filename = os.path.join(temp_dir, "X_test.dat")
                    y_test_filename = os.path.join(temp_dir, "y_test.dat")

                    X_train_mmap = np.memmap(
                        X_train_filename, dtype="float32", mode="w+", shape=X_train.shape
                    )
                    y_train_mmap = np.memmap(
                        y_train_filename, dtype="bool", mode="w+", shape=y_train.shape
                    )
                    X_test_mmap = np.memmap(
                        X_test_filename, dtype="float32", mode="w+", shape=X_test.shape
                    )
                    y_test_mmap = np.memmap(
                        y_test_filename, dtype="bool", mode="w+", shape=y_test.shape
                    )

                    X_train_mmap[:] = X_train[:]
                    y_train_mmap[:] = y_train[:]
                    X_test_mmap[:] = X_test[:]
                    y_test_mmap[:] = y_test[:]

                    X_train_mmap.flush()
                    y_train_mmap.flush()
                    X_test_mmap.flush()
                    y_test_mmap.flush()

                    shape_train = X_train.shape
                    shape_test = X_test.shape
                    del X_train, y_train, X_test, y_test
                    run_func = functools.partial(
                        run_logistic_regression_on_latent,
                        X_train_filename=X_train_filename,
                        y_train_filename=y_train_filename,
                        X_test_filename=X_test_filename,
                        y_test_filename=y_test_filename,
                        shape_train=shape_train,
                        shape_test=shape_test,
                    )

                    with Pool() as pool:
                        res_rows = list(
                            tqdm(
                                pool.imap(run_func, range(sae_dim)),
                                total=sae_dim,
                                desc=(
                                    "Logistic regression on each latent dimension for "
                                    f"{annotation.name}: {class_name}"
                                ),
                            )
                        )

                res_df = pd.DataFrame(
                    res_rows, columns=["dim", "precision", "recall", "f1"]
                ).sort_values(by="f1", ascending=False)
                logger.info(f"Results: {res_df.head()}")

                res_df.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")

            del seq_to_annotation_entries, examples, train_examples, test_examples
            gc.collect()


if __name__ == "__main__":
    single_latent()
