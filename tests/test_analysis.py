"""Tests for the spike prominence and width numerics."""

import unittest

import numpy as np

from aind_ophys_movie_qc_library.analysis import get_spike_prominence_and_width

FRAME_RATE = 10.0
SPIKE_THRESHOLD = 0.05


class TestSpikeProminenceAndWidth(unittest.TestCase):
    """Prominence and width for isolated and adjacent spikes."""

    def test_isolated_spike_prominence_is_peak_above_baseline(self) -> None:
        """A single spike on a flat baseline has prominence equal to its height."""
        signal = np.array([0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)
        spikes = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)

        prominences, widths = get_spike_prominence_and_width(
            signal, spikes, FRAME_RATE, SPIKE_THRESHOLD
        )

        self.assertEqual(prominences.shape, (1,))
        self.assertAlmostEqual(float(prominences[0]), 4.0)
        self.assertEqual(widths.shape, (1,))
        self.assertAlmostEqual(float(widths[0]), 2.0 / FRAME_RATE)

    def test_adjacent_larger_spike_yields_negative_prominence(self) -> None:
        """A spike immediately right of a larger one scores below zero.

        ``pre_spikes`` is then the single larger neighbour, so the local
        minimum exceeds the spike's own amplitude.
        """
        signal = np.array([0.0, 0.1, 5.0, 3.0, 0.2, 0.1, 0.0], dtype=float)
        spikes = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

        prominences, _ = get_spike_prominence_and_width(
            signal, spikes, FRAME_RATE, SPIKE_THRESHOLD
        )

        self.assertAlmostEqual(float(prominences[0]), 5.0)
        self.assertLess(float(prominences[1]), 0.0)

    def test_half_max_crossing_at_first_frame_is_honoured(self) -> None:
        """A crossing at offset 0 gives a zero-length post-peak width.

        The guards test ``.size == 0`` rather than truthiness: here
        ``post_half_idxs`` is exactly ``[0]``, which is a real crossing at the
        first frame, not an empty result. Truthiness would instead fall back to
        the full window and report a width of 4 frames for this spike.
        """
        signal = np.array([0.0, 0.1, 5.0, 3.0, 4.5, 4.6, 4.7], dtype=float)
        spikes = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

        _, widths = get_spike_prominence_and_width(
            signal, spikes, FRAME_RATE, SPIKE_THRESHOLD
        )

        # pre_half_idxs is empty for this spike, contributing len(pre_spikes)
        # of 1 frame; the post-peak half-width contributes 0.
        self.assertAlmostEqual(float(widths[1]), 1.0 / FRAME_RATE)


if __name__ == "__main__":
    unittest.main()
