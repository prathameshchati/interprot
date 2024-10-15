import unittest
from unittest.mock import Mock, patch

import pandas as pd

from plm_interpretability.logistic_regression_probe.annotations import ResidueAnnotation
from plm_interpretability.logistic_regression_probe.utils import (
    get_annotation_entries_for_class,
    make_examples_from_annotation_entries,
)


class TestUtils(unittest.TestCase):
    @patch("plm_interpretability.logistic_regression_probe.utils.get_sae_acts")
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
