import functools
import logging
import os
import random
import tempfile
import warnings
from dataclasses import dataclass
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

from plm_interpretability.sae_model import SparseAutoencoder
from plm_interpretability.utils import get_layer_activations, parse_swissprot_annotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ResidueAnnotation:
    name: str
    swissprot_header: str
    class_names: list[str]

    # Class name used to indicate that we don't care about the annotation class,
    # as long as the annotation exists. E.g. signal peptides annotations look like
    # `{'start': 1, 'end': 24, 'evidence': 'ECO:0000255'}`, so we just classify
    # whether the residue is part of a signal peptide or not.
    ALL_CLASSES = "all"


RESIDUE_ANNOTATIONS = [
    ResidueAnnotation(
        name="DNA binding",
        swissprot_header="DNA_BIND",
        class_names=["H-T-H motif", "Homeobox", "Nuclear receptor", "HMG box"],
    ),
    ResidueAnnotation(
        name="Motif",
        swissprot_header="MOTIF",
        class_names=[
            "Nuclear localization signal",
            "Nuclear export signal",
            "DEAD box",
            "Cell attachment site",
            "JAMM motif",
            "SH3-binding",
            "Cysteine switch",
        ],
    ),
    ResidueAnnotation(
        name="Topological domain",
        swissprot_header="TOPO_DOM",
        class_names=[
            "Cytoplasmic",
            "Extracellular",
            "Lumenal",
            "Periplasmic",
            "Mitochondrial intermembrane",
            "Mitochondrial matrix",
            "Virion surface",
            "Intravirion",
        ],
    ),
    ResidueAnnotation(
        name="Domain [FT]",
        swissprot_header="DOMAIN",
        class_names=[
            "Protein kinase",
            "tr-type G",
            "Radical SAM core",
            "ABC transporter",
            "Helicase ATP-binding",
            "Glutamine amidotransferase type-1",
            "ATP-grasp",
            "S4 RNA-binding",
        ],
    ),
    ResidueAnnotation(
        name="Active site",
        swissprot_header="ACT_SITE",
        class_names=[
            "Proton acceptor",
            "Proton donor",
            "Nucleophile",
            "Charge relay system",
        ],
    ),
    ResidueAnnotation(
        name="Signal peptide",
        swissprot_header="SIGNAL",
        class_names=[ResidueAnnotation.ALL_CLASSES],
    ),
    ResidueAnnotation(
        name="Transit peptide",
        swissprot_header="TRANSIT",
        class_names=[ResidueAnnotation.ALL_CLASSES, "Mitochondrion", "Chloroplast"],
    ),
]


@functools.lru_cache(maxsize=10000)
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
    for _, row in swissprot_df[swissprot_df[annotation.name].notna()].iterrows():
        seq = row["Sequence"]
        entries = parse_swissprot_annotation(
            row[annotation.name], header=annotation.swissprot_header
        )
        if class_name != ResidueAnnotation.ALL_CLASSES:
            # The note field is sometimes like "Homeobox", "Homeobox 1", etc.,
            # so use string `in` to check.
            entries = [e for e in entries if class_name in e.get("note", "")]
        if len(entries) > 0:
            seq_to_annotation_entries[seq] = entries
    logger.info(f"Found {len(seq_to_annotation_entries)} sequences with class {class_name}")

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


def run_logistic_regression_on_latent(args):
    (
        dim,
        X_train_filename,
        y_train_filename,
        X_test_filename,
        y_test_filename,
        shape_train,
        shape_test,
    ) = args

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
def latent_probe(
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
                temp_dir = tempfile.mkdtemp()
                with tempfile.TemporaryDirectory() as temp_dir:
                    X_train_filename = os.path.join(temp_dir, "X_train.dat")
                    y_train_filename = os.path.join(temp_dir, "y_train.dat")
                    X_test_filename = os.path.join(temp_dir, "X_test.dat")
                    y_test_filename = os.path.join(temp_dir, "y_test.dat")

                    with (
                        np.memmap(
                            X_train_filename,
                            dtype="float32",
                            mode="w+",
                            shape=X_train.shape,
                        ) as X_train_mmap,
                        np.memmap(
                            y_train_filename,
                            dtype="bool",
                            mode="w+",
                            shape=y_train.shape,
                        ) as y_train_mmap,
                        np.memmap(
                            X_test_filename,
                            dtype="float32",
                            mode="w+",
                            shape=X_test.shape,
                        ) as X_test_mmap,
                        np.memmap(
                            y_test_filename, dtype="bool", mode="w+", shape=y_test.shape
                        ) as y_test_mmap,
                    ):
                        X_train_mmap[:] = X_train[:]
                        y_train_mmap[:] = y_train[:]
                        X_test_mmap[:] = X_test[:]
                        y_test_mmap[:] = y_test[:]

                        X_train_mmap.flush()
                        y_train_mmap.flush()
                        X_test_mmap.flush()
                        y_test_mmap.flush()

                        # Create arguments for each process
                        args = [
                            (
                                dim,
                                X_train_filename,
                                y_train_filename,
                                X_test_filename,
                                y_test_filename,
                                X_train.shape,
                                X_test.shape,
                            )
                            for dim in range(sae_dim)
                        ]

                        with Pool() as pool:
                            res_rows = list(
                                tqdm(
                                    pool.imap(run_logistic_regression_on_latent, args),
                                    total=sae_dim,
                                    desc=(
                                        "Running logistic regression on each latent for "
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


if __name__ == "__main__":
    latent_probe()
