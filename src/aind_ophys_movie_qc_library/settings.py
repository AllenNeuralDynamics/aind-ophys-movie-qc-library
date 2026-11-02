"""CLI/environment settings for registered-movie quality control."""

import json
from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pixel value at which a ScanImage detector is considered saturated.
SCANIMAGE_MAXIMUM_PIXEL_BRIGHTNESS_LEVEL = 18423

# Pixels trimmed from each edge before any statistic is computed.
DEFAULT_CROP = [30, 30]


class MovieQCSettings(BaseSettings, cli_parse_args=True):
    """Settings for registered-movie QC using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MOVIE_QC_",
        case_sensitive=False,
        extra="ignore",
    )

    input_dir: Path = Field(
        default=Path("../data/"),
        description=(
            "Directory containing the motion-correction results and the raw "
            "asset metadata. Searched recursively for the registered movie."
        ),
    )
    output_dir: Path = Field(
        default=Path("../results/"),
        description="Directory where output files will be saved.",
    )
    start_frame: int = Field(
        default=1,
        description="First frame of the movie block used for the main analysis.",
    )
    end_frame: int = Field(
        default=10000,
        description="End of the movie block used for the main analysis (exclusive).",
    )
    crop: Optional[list] = Field(
        default=None,
        description=(
            "Pixels to remove from each edge before analysis, as the JSON "
            "list '[px_y, px_x]'. Useful to drop motion-correction edge "
            "artefacts. Defaults to [30, 30]."
        ),
    )
    max_pixel_range: int = Field(
        default=SCANIMAGE_MAXIMUM_PIXEL_BRIGHTNESS_LEVEL,
        description="Pixel value at or above which saturation is counted.",
    )
    min_pixel_range: int = Field(
        default=0,
        description="Pixel value at or below which under-saturation is counted.",
    )
    decay_time: float = Field(
        default=0.2,
        description=(
            "Estimated half decay time of the event reporter, in seconds. "
            "The default is for GCaMP6f."
        ),
    )
    frame_rate: Optional[float] = Field(
        default=None,
        description=(
            "Frame rate in Hz, used only when it cannot be resolved from the "
            "registered movie or the acquisition metadata. It scales the "
            "epilepsy event widths and every per-second photon statistic, so "
            "it is never defaulted silently."
        ),
    )
    dff_single_event_size: float = Field(
        default=0.15,
        description=(
            "Estimated dF/F event size of the event reporter for single "
            "events, in arbitrary units. The default is for GCaMP6f."
        ),
    )

    @field_validator("crop", mode="before")
    @classmethod
    def _coerce_crop(cls, value: Any) -> List[int]:
        """Normalise ``crop`` to a two-element list of ints.

        Declared ``Optional[list]`` rather than ``Tuple[int, int]`` or with a
        ``default_factory``: ``auto-app-panel`` renders a ``List[int]`` field
        as a single integer box (which cannot express a pair) and crashes on a
        ``default_factory`` that returns an empty list. A JSON string is the
        only form a Code Ocean panel can emit for a pair, so it is accepted
        here.

        Parameters
        ----------
        value : Any
            ``None``, a JSON string such as ``"[30, 30]"``, or a
            two-element sequence.

        Returns
        -------
        list of int
            The two crop values.

        Raises
        ------
        ValueError
            If the value does not resolve to exactly two integers.
        """
        if value is None:
            return list(DEFAULT_CROP)
        if isinstance(value, str):
            value = json.loads(value)
        coerced = [int(item) for item in value]
        if len(coerced) != 2:
            raise ValueError(
                "crop must have exactly two values, '[px_y, px_x]'; got " f"{coerced!r}"
            )
        return coerced
