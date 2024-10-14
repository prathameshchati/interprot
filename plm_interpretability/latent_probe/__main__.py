import logging
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache

import click
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EsmModel

from plm_interpretability.sae_model import SparseAutoencoder
from plm_interpretability.utils import get_layer_activations, parse_swissprot_annotation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ResidueAnnotation:
    name: str
    swissprot_header: str
    values: list[str]


RESIDUE_ANNOTATIONS = [
    ResidueAnnotation(
        name="DNA binding",
        swissprot_header="DNA_BIND",
        values=["H-T-H motif", "Homeobox", "Nuclear receptor", "HMG box"],
    ),
    ResidueAnnotation(
        name="Motif",
        swissprot_header="MOTIF",
        values=[
            "Nuclear localization signal",
            "Nuclear export signal",
            "DEAD box",
            "Cell attachment site",
            "JAMM motif",
            "SH3-binding",
            "Cysteine switch",
        ],
    ),
]


@lru_cache(maxsize=10000)
def get_sae_acts(
    seq: str,
    tokenizer: AutoTokenizer,
    plm_model: EsmModel,
    sae_model: SparseAutoencoder,
    plm_layer: int,
):
    esm_layer_acts = get_layer_activations(
        tokenizer=tokenizer, plm=plm_model, seqs=[seq], layer=plm_layer
    )
    return sae_model.get_acts(esm_layer_acts)[1:-1]


@click.command()
@click.option(
    "--sae-checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to the SAE checkpoint file",
)
@click.option(
    "--sae-dim", type=int, required=True, help="Dimension of the sparse autoencoder"
)
@click.option(
    "--plm-dim", type=int, required=True, help="Dimension of the protein language model"
)
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
def latent_probe(
    sae_checkpoint: str,
    sae_dim: int,
    plm_dim: int,
    plm_layer: int,
    swissprot_tsv: str,
    output_dir: str,
):
    """
    Run 1D logistic regression probing for each latent dimension for SAE evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    # Load pLM and SAE
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    plm_model = (
        EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval()
    )

    sae_model = SparseAutoencoder(plm_dim, sae_dim).to(device)
    sae_model.load_state_dict(torch.load(sae_checkpoint, map_location=device))

    df = pd.read_csv(swissprot_tsv, sep="\t")

    for annotation in RESIDUE_ANNOTATIONS:
        logger.info(f"Processing annotation: {annotation.name}")
        os.makedirs(os.path.join(output_dir, annotation.name), exist_ok=True)

        for label in annotation.values:
            output_path = os.path.join(output_dir, annotation.name, f"{label}.csv")
            if os.path.exists(output_path):
                logger.warning(f"Skipping {output_path} because it already exists")
                continue

            seq_to_annotation_entries = {}
            for _, row in df[df[annotation.name].notna()].iterrows():
                seq = row["Sequence"]
                entries = parse_swissprot_annotation(
                    row[annotation.name], header=annotation.swissprot_header
                )
                entries = [e for e in entries if label in e.get("note", "")]
                if len(entries) > 0:
                    seq_to_annotation_entries[seq] = entries
            logger.info(
                f"Found {len(seq_to_annotation_entries)} sequences with label {label}"
            )

            # Collect examples: 1 example per residue where the label is whether the
            # residue is positive for the label.
            examples = []
            num_positive_examples = 0
            for seq, entries in seq_to_annotation_entries.items():
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
                            "label": i in positive_positions,
                        }
                    )
                    if i in positive_positions:
                        num_positive_examples += 1

            logger.info(
                f"{num_positive_examples}/{len(examples)} positive/total examples"
            )

            train_examples, test_examples = train_test_split(
                examples,
                test_size=0.1,
                random_state=42,
                stratify=[e["label"] for e in examples],
            )

            with warnings.catch_warnings():
                # LogisticRegression throws warnings when it can't converge.
                # This is expected for most dimensions.
                warnings.simplefilter("ignore")

                res_rows = []
                for dim in range(sae_dim):
                    model = LogisticRegression(class_weight="balanced")
                    X_train = [[e["sae_acts"][dim]] for e in train_examples]
                    y_train = [e["label"] for e in train_examples]
                    X_test = [[e["sae_acts"][dim]] for e in test_examples]
                    y_test = [e["label"] for e in test_examples]

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    res_rows.append([dim, precision, recall, f1])

                res_df = pd.DataFrame(
                    res_rows, columns=["dim", "precision", "recall", "f1"]
                ).sort_values(by="f1", ascending=False)
                logger.info(f"Results: {res_df.head()}")

                res_df.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    latent_probe()
