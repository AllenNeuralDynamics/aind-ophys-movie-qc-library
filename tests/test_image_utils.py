"""Tests for the combined ROI + photon QC figure helpers.

The composite is the ``reference`` of the photon-detection metric, so a silent
failure here leaves a reviewer with a broken link.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Tuple

from PIL import Image

from aind_ophys_movie_qc_library.utils import image_utils


def _write_png(path: Path, size: Tuple[int, int]) -> Path:
    """Write a solid-colour PNG of the given size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


class TestCombineImagesVertically(unittest.TestCase):
    """Stacking, padding and the empty-input guard."""

    def test_dimensions_include_padding(self) -> None:
        """Height is the sum plus one padding gap; width is the maximum."""
        images = [Image.new("RGB", (40, 10)), Image.new("RGB", (20, 30))]
        combined = image_utils.combine_images_vertically(images, padding=5)
        self.assertEqual(combined.size, (40, 45))

    def test_single_image_has_no_padding(self) -> None:
        """One image means zero gaps."""
        combined = image_utils.combine_images_vertically([Image.new("RGB", (8, 9))], padding=7)
        self.assertEqual(combined.size, (8, 9))

    def test_empty_input_raises(self) -> None:
        """max() on an empty sequence would raise something unhelpful."""
        with self.assertRaises(ValueError):
            image_utils.combine_images_vertically([])


class TestSaveImage(unittest.TestCase):
    """The writer creates its parent directory."""

    def test_creates_parents(self) -> None:
        """A nested destination is created rather than raising."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "VISp_0" / "movie_qc" / "combined.png"
            written = image_utils.save_image(Image.new("RGB", (4, 4)), target)
            self.assertTrue(written.is_file())


class TestBuildCombinedRoiPhotonImage(unittest.TestCase):
    """The composite figure the photon metric references."""

    def test_stacks_the_two_panels(self) -> None:
        """The output is as tall as both panels plus the default padding."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmentation = _write_png(root / "segmentation.png", (30, 20))
            poisson = _write_png(root / "poisson.png", (30, 25))
            output = root / "out" / "combined.png"
            written = image_utils.build_combined_roi_photon_image(segmentation, poisson, output)
            with Image.open(written) as combined:
                self.assertEqual(combined.size, (30, 55))

    def test_missing_panel_names_the_absent_file(self) -> None:
        """A missing panel is named before PIL is ever reached."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmentation = _write_png(root / "segmentation.png", (10, 10))
            missing = root / "poisson.png"
            with self.assertRaises(FileNotFoundError) as ctx:
                image_utils.build_combined_roi_photon_image(
                    segmentation, missing, root / "combined.png"
                )
        self.assertIn("poisson.png", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
