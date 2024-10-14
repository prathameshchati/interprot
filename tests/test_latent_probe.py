import unittest
from unittest.mock import Mock, patch

from plm_interpretability.latent_probe.__main__ import (
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
