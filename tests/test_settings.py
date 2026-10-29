"""Tests for MovieQCSettings, including the app-panel drivability contract.

A Code Ocean app panel can only emit ``--name=value``. Two properties must
therefore hold for every settings field, and are asserted here rather than
trusted to the generator: the field parses in that form, and its name is
longer than one character (pydantic-settings renders a one-character field as
a short option, which the panel can never produce).
"""

import sys
import typing
import unittest
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import patch

from pydantic import ValidationError

from aind_ophys_movie_qc_library.settings import (
    DEFAULT_CROP,
    SCANIMAGE_MAXIMUM_PIXEL_BRIGHTNESS_LEVEL,
    MovieQCSettings,
)


def _make_settings(
    argv: Optional[List[str]] = None, **kwargs: object
) -> MovieQCSettings:
    """Construct MovieQCSettings with a controlled sys.argv.

    ``cli_parse_args=True`` otherwise consumes the test runner's own
    arguments.
    """
    with patch.object(sys, "argv", ["run_capsule.py"] + list(argv or [])):
        return MovieQCSettings(**kwargs)


# Per field type: a ``--name=value`` payload and the value it must parse to.
CLI_SAMPLES = {
    int: ("2", 2),
    float: ("1.5", 1.5),
    Path: ("/tmp/movie-qc-test", Path("/tmp/movie-qc-test")),
    list: ("[10, 20]", [10, 20]),
}


def _cli_sample(annotation: object) -> Tuple[str, object]:
    """Return the CLI payload and expected parsed value for an annotation.

    ``Optional[X]`` is driven as ``X``. An annotation with no sample raises
    ``KeyError``, so a new field type cannot pass the contract test unseen.
    """
    if typing.get_origin(annotation) is typing.Union:
        annotation = next(a for a in typing.get_args(annotation) if a is not type(None))
    return CLI_SAMPLES[annotation]


class TestMovieQCSettingsDefaults(unittest.TestCase):
    """Field defaults and command-line parsing."""

    def test_defaults(self) -> None:
        """Every field carries its documented default."""
        settings = _make_settings()
        self.assertEqual(settings.input_dir, Path("../data/"))
        self.assertEqual(settings.output_dir, Path("../results/"))
        self.assertEqual(settings.start_frame, 1)
        self.assertEqual(settings.end_frame, 10000)
        self.assertEqual(settings.crop, DEFAULT_CROP)
        self.assertEqual(settings.max_pixel_range, SCANIMAGE_MAXIMUM_PIXEL_BRIGHTNESS_LEVEL)
        self.assertEqual(settings.min_pixel_range, 0)
        self.assertEqual(settings.decay_time, 0.2)
        self.assertEqual(settings.dff_single_event_size, 0.15)

    def test_frame_rate_sentinel_is_none_not_zero(self) -> None:
        """An unset frame rate is ``None``, never a ``0`` sentinel."""
        self.assertIsNone(_make_settings().frame_rate)

    def test_frame_rate_override_parses(self) -> None:
        """An explicit frame rate is still available as a last resort."""
        self.assertEqual(_make_settings(["--frame_rate=11.5"]).frame_rate, 11.5)


class TestCropCoercion(unittest.TestCase):
    """``crop`` is the one field with an app-panel-shaped hazard."""

    def test_none_yields_the_default_pair(self) -> None:
        """An unset crop resolves to the default pair."""
        self.assertEqual(_make_settings(crop=None).crop, DEFAULT_CROP)

    def test_json_string_is_parsed(self) -> None:
        """A JSON string is the only pair form a CO panel can emit."""
        self.assertEqual(_make_settings(["--crop=[10, 20]"]).crop, [10, 20])

    def test_json_string_reaches_the_validator_directly(self) -> None:
        """An env-supplied value arrives as a raw string, not pre-parsed."""
        self.assertEqual(_make_settings(crop="[12, 34]").crop, [12, 34])

    def test_sequence_is_coerced_to_ints(self) -> None:
        """A tuple of floats is accepted and coerced."""
        self.assertEqual(_make_settings(crop=(4.0, 5.0)).crop, [4, 5])

    def test_wrong_length_is_rejected(self) -> None:
        """Consumption is positional, so exactly two values are required."""
        with self.assertRaises(ValidationError):
            _make_settings(crop=[1, 2, 3])


class TestAppPanelContract(unittest.TestCase):
    """Every field must be drivable by a Code Ocean app panel."""

    def test_no_single_character_field_names(self) -> None:
        """No field name is one character long."""
        offenders = [name for name in MovieQCSettings.model_fields if len(name) == 1]
        self.assertEqual(offenders, [])

    def test_every_field_parses_as_named_argument(self) -> None:
        """Each field parses the ``--name=value`` form the panel emits."""
        for name, field in MovieQCSettings.model_fields.items():
            with self.subTest(field=name):
                payload, expected = _cli_sample(field.annotation)
                settings = _make_settings([f"--{name}={payload}"])
                self.assertEqual(getattr(settings, name), expected)

    def test_every_field_has_a_description(self) -> None:
        """A panel with no descriptions is unusable by an operator."""
        missing = [
            name for name, field in MovieQCSettings.model_fields.items() if not field.description
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
