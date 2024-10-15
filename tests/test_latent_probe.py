import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from click.testing import CliRunner

from plm_interpretability.latent_probe.__main__ import (
    ResidueAnnotation,
    get_annotation_entries_for_class,
    latent_probe,
    make_examples_from_annotation_entries,
)


class TestLatentProbe(unittest.TestCase):
    @patch("plm_interpretability.latent_probe.__main__.get_sae_acts")
    def test_make_examples_from_annotation_entries(self, mock_get_sae_acts):
        # Mock the necessary objects
        mock_tokenizer = Mock()
        mock_plm_model = Mock()
        mock_sae_model = Mock()

        # Set up test data
        seq_to_annotation_entries = {
            "ABCDEF": [
                {"start": 2, "end": 4},
            ],
            "GHIJKL": [
                {"start": 1, "end": 2},
                {"start": 5, "end": 5},
            ],
        }

        mock_get_sae_acts.side_effect = [
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
                [1.1, 1.2],
            ],  # For "ABCDEF"
            [
                [1.3, 1.4],
                [1.5, 1.6],
                [1.7, 1.8],
                [1.9, 2.0],
                [2.1, 2.2],
                [2.3, 2.4],
            ],  # For "GHIJKL"
        ]

        examples = make_examples_from_annotation_entries(
            seq_to_annotation_entries,
            mock_tokenizer,
            mock_plm_model,
            mock_sae_model,
            plm_layer=24,
        )

        self.assertEqual(len(examples), 12)
        self.assertEqual(examples[0], {"sae_acts": [0.1, 0.2], "target": False})
        self.assertEqual(examples[1], {"sae_acts": [0.3, 0.4], "target": True})
        self.assertEqual(examples[2], {"sae_acts": [0.5, 0.6], "target": True})
        self.assertEqual(examples[3], {"sae_acts": [0.7, 0.8], "target": True})

        self.assertEqual(examples[6], {"sae_acts": [1.3, 1.4], "target": True})
        self.assertEqual(examples[7], {"sae_acts": [1.5, 1.6], "target": True})
        self.assertEqual(examples[8], {"sae_acts": [1.7, 1.8], "target": False})
        self.assertEqual(examples[9], {"sae_acts": [1.9, 2.0], "target": False})
        self.assertEqual(examples[10], {"sae_acts": [2.1, 2.2], "target": True})

        mock_get_sae_acts.assert_any_call(
            seq="ABCDEF",
            tokenizer=mock_tokenizer,
            plm_model=mock_plm_model,
            sae_model=mock_sae_model,
            plm_layer=24,
        )
        mock_get_sae_acts.assert_any_call(
            seq="GHIJKL",
            tokenizer=mock_tokenizer,
            plm_model=mock_plm_model,
            sae_model=mock_sae_model,
            plm_layer=24,
        )

    def test_get_annotation_entries_for_class(self):
        mock_df = pd.DataFrame(
            {
                "Sequence": ["ABCDEF", "GHIJKL", "MNOPQR"],
                "DNA binding": [
                    'DNA_BIND 1..3; /note="H-T-H motif"',
                    'DNA_BIND 2..4; /note="Homeobox"',
                    'DNA_BIND 1..6; /note="Nuclear receptor"',
                ],
            }
        )

        annotation = ResidueAnnotation(
            name="DNA binding",
            swissprot_header="DNA_BIND",
            class_names=["H-T-H motif", "Homeobox", "Nuclear receptor"],
        )

        result = get_annotation_entries_for_class(
            mock_df, annotation, "H-T-H motif", max_seqs_per_task=10
        )
        self.assertEqual(len(result), 1)
        self.assertIn("ABCDEF", result)
        self.assertEqual(result["ABCDEF"], [{"start": 1, "end": 3, "note": "H-T-H motif"}])

        result = get_annotation_entries_for_class(
            mock_df, annotation, "Homeobox", max_seqs_per_task=10
        )
        self.assertEqual(len(result), 1)
        self.assertIn("GHIJKL", result)
        self.assertEqual(result["GHIJKL"], [{"start": 2, "end": 4, "note": "Homeobox"}])

        result = get_annotation_entries_for_class(
            mock_df, annotation, ResidueAnnotation.ALL_CLASSES, max_seqs_per_task=10
        )
        self.assertEqual(len(result), 3)
        self.assertIn("ABCDEF", result)
        self.assertIn("GHIJKL", result)
        self.assertIn("MNOPQR", result)

        result = get_annotation_entries_for_class(
            mock_df, annotation, ResidueAnnotation.ALL_CLASSES, max_seqs_per_task=2
        )
        self.assertEqual(len(result), 2)

        result = get_annotation_entries_for_class(
            mock_df, annotation, "Non-existent", max_seqs_per_task=10
        )
        self.assertEqual(len(result), 0)

    @patch("plm_interpretability.latent_probe.__main__.get_annotation_entries_for_class")
    @patch("plm_interpretability.latent_probe.__main__.torch.load")
    @patch("plm_interpretability.latent_probe.__main__.AutoTokenizer.from_pretrained")
    @patch("plm_interpretability.latent_probe.__main__.EsmModel.from_pretrained")
    @patch("plm_interpretability.latent_probe.__main__.SparseAutoencoder")
    @patch("plm_interpretability.latent_probe.__main__.get_sae_acts")
    def test_latent_probe_e2e(
        self,
        mock_get_sae_acts,
        mock_sae,
        mock_esm,
        mock_tokenizer,
        mock_torch_load,
        mock_get_annotation_entries_for_class,
    ):
        mock_torch_load.return_value = {}
        mock_tokenizer.return_value = None
        mock_esm.return_value = Mock(to=Mock())
        mock_sae.return_value = Mock(to=Mock())

        # Assign some random positions our test annotation
        mock_get_annotation_entries_for_class.return_value = {
            "AAAAAAAAAA": [
                {"start": 1, "end": 5, "note": "H-T-H motif"},
                {"start": 7, "end": 7, "note": "H-T-H motif"},
            ],
            "CCCCCCCCCC": [
                {"start": 2, "end": 5, "note": "H-T-H motif"},
            ],
        }

        # Mock SAE activations to make hidden dim 2 correlate perfectly with the
        # test annotations
        def mock_sae_acts(seq, *args, **kwargs):
            acts = np.zeros((len(seq), 10))
            if seq == "AAAAAAAAAA":
                acts[0:5, 2] = 1
                acts[6, 2] = 1
            elif seq == "CCCCCCCCCC":
                acts[1:5, 2] = 1
            return acts

        mock_get_sae_acts.side_effect = mock_sae_acts

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("dummy.pt", "w") as f:
                f.write("dummy checkpoint")
            with open("dummy.tsv", "w") as f:
                f.write("dummy swissprot data")

            result = runner.invoke(
                latent_probe,
                [
                    "--sae-checkpoint",
                    "dummy.pt",
                    "--sae-dim",
                    "10",
                    "--plm-dim",
                    "1280",
                    "--plm-layer",
                    "24",
                    "--swissprot-tsv",
                    "dummy.tsv",
                    "--output-dir",
                    "dummy_output",
                    "--annotation-names",
                    "DNA binding",
                    "--max-seqs-per-task",
                    "4",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            results_df = pd.read_csv("dummy_output/DNA binding/H-T-H motif.csv")

        # Check that hidden dim 2 has perfect precision, recall, and F1,
        # whereas the rest of the dims should have 0.
        for _, row in results_df.iterrows():
            if row["dim"] == 2:
                self.assertEqual(row["precision"], 1)
                self.assertEqual(row["recall"], 1)
                self.assertEqual(row["f1"], 1)
            else:
                self.assertEqual(row["precision"], 0)
                self.assertEqual(row["recall"], 0)
                self.assertEqual(row["f1"], 0)
