"""PIL helpers for the combined ROI + photon QC figure.

Kept out of ``qc.py`` so metric construction stays free of image I/O, and out
of ``analysis.py`` so it is cheap to test: the composite figure is the
``reference`` of the photon-detection metric, so a silent failure here leaves
a reviewer with a broken link.
"""

from pathlib import Path
from typing import List

from PIL import Image

WHITE = (255, 255, 255)


def combine_images_vertically(
    images: List[Image.Image],
    padding: int = 10,
) -> Image.Image:
    """Stack images vertically on a white background.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Images to combine; must be non-empty.
    padding : int, optional
        Pixels of white space between adjacent images.

    Returns
    -------
    PIL.Image.Image
        The combined image.

    Raises
    ------
    ValueError
        If ``images`` is empty.
    """
    if not images:
        raise ValueError("combine_images_vertically needs at least one image")
    total_height = sum(image.height for image in images) + (len(images) - 1) * padding
    max_width = max(image.width for image in images)

    combined = Image.new("RGB", (max_width, total_height), WHITE)
    offset = 0
    for image in images:
        combined.paste(image, (0, offset))
        offset += image.height + padding
    return combined


def save_image(image: Image.Image, output_path: Path) -> Path:
    """Write an image, creating its parent directory.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to save.
    output_path : Path
        Destination path.

    Returns
    -------
    Path
        The path written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def build_combined_roi_photon_image(
    segmentation_png: Path,
    poisson_png: Path,
    output_path: Path,
) -> Path:
    """Stack the segmentation and Poisson figures into one reviewer figure.

    Parameters
    ----------
    segmentation_png : Path
        Basic-segmentation overlay figure.
    poisson_png : Path
        Variance-versus-mean Poisson figure.
    output_path : Path
        Destination for the combined figure.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    FileNotFoundError
        If either input figure is missing, naming the one that is absent.
    """
    for panel in (segmentation_png, poisson_png):
        if not Path(panel).is_file():
            raise FileNotFoundError(
                f"Cannot build the combined ROI/photon figure: {panel} does not exist."
            )
    with Image.open(segmentation_png) as segmentation, Image.open(poisson_png) as poisson:
        combined = combine_images_vertically([segmentation, poisson])
    return save_image(combined, output_path)
