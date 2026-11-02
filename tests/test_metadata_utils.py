"""Tests for the metadata view, the frame-rate chain and the Code block."""

import logging
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from typing import Optional, Tuple

from aind_pophys_metadata import CoreMetadata

from aind_ophys_movie_qc_library.utils import metadata_utils as mu
from tests import fixtures


def core_from(**files: dict) -> CoreMetadata:
    """Load metadata written into a throwaway directory.

    Goes through the real :meth:`aind_pophys_metadata.CoreMetadata.load` path
    rather than constructing the dataclass, so the tests exercise the readers
    this library actually depends on.

    Parameters
    ----------
    **files
        File name (with ``.`` replaced by ``_``) to JSON payload, e.g.
        ``acquisition_json=...``.

    Returns
    -------
    aind_pophys_metadata.CoreMetadata
        The located metadata the readers consume.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for key, payload in files.items():
            fixtures.write_json(root, key.replace("_json", ".json"), payload)
        return CoreMetadata.load(root)


def acquisition_core(**kwargs: object) -> CoreMetadata:
    """Load an acquisition fixture.

    Parameters
    ----------
    **kwargs
        Forwarded to :func:`tests.fixtures.v2_acquisition`.

    Returns
    -------
    aind_pophys_metadata.CoreMetadata
        The located metadata the readers consume.
    """
    return core_from(acquisition_json=fixtures.v2_acquisition(**kwargs))


class TestBuildMovieQCMetadata(unittest.TestCase):
    """The view maps each shared field, and requires none of them."""

    def test_maps_the_shared_fields(self) -> None:
        """Each view field comes from its shared reader."""
        core = core_from(
            acquisition_json=fixtures.v2_acquisition(),
            data_description_json=fixtures.data_description(),
        )
        metadata = mu.build_movie_qc_metadata(core)
        self.assertEqual(metadata.frame_rate, fixtures.FRAME_RATE)
        self.assertEqual(metadata.fov_scale_factor, fixtures.FOV_SCALE_FACTOR)
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)
        self.assertEqual(metadata.dataset_name, fixtures.DATASET_NAME)

    def test_nothing_is_required(self) -> None:
        """A core file with none of the optional content still yields a view."""
        core = acquisition_core(frame_rate=None, subject_id=None, fov_scale_factor=None)
        metadata = mu.build_movie_qc_metadata(core)
        self.assertIsNone(metadata.frame_rate)
        self.assertIsNone(metadata.fov_scale_factor)
        self.assertIsNone(metadata.subject_id)
        self.assertIsNone(metadata.dataset_name)

    def test_metadata_view_is_hashable(self) -> None:
        """A frozen dataclass with a mutable field silently loses this."""
        self.assertIsInstance(hash(mu.MovieQCMetadata(frame_rate=1.0)), int)


class TestReadMovieFrameRate(unittest.TestCase):
    """The rate motion correction recorded on the movie, and loud misses."""

    def test_reads_the_recorded_rate(self) -> None:
        """frame_rate_hz is decoded from the movie's JSON metadata dataset."""
        with tempfile.TemporaryDirectory() as tmp:
            movie = fixtures.write_registered_movie(Path(tmp))
            rate = mu.read_movie_frame_rate(movie)
        self.assertEqual(rate, fixtures.UPSTREAM_FRAME_RATE)

    def test_missing_metadata_dataset_warns_naming_it(self) -> None:
        """A movie without the dataset is reported, then the chain continues."""
        with tempfile.TemporaryDirectory() as tmp:
            movie = fixtures.write_registered_movie(Path(tmp), with_metadata=False)
            with self.assertLogs(mu.logger, level=logging.WARNING) as logs:
                rate = mu.read_movie_frame_rate(movie)
        self.assertIsNone(rate)
        self.assertIn(mu.MOVIE_METADATA_DATASET, "".join(logs.output))
        self.assertIn(movie.name, "".join(logs.output))

    def test_missing_key_warns_naming_it(self) -> None:
        """A present dataset without the key is never silently skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            movie = fixtures.write_registered_movie(Path(tmp), frame_rate=None)
            with self.assertLogs(mu.logger, level=logging.WARNING) as logs:
                rate = mu.read_movie_frame_rate(movie)
        self.assertIsNone(rate)
        self.assertIn(mu.MOVIE_FRAME_RATE_KEY, "".join(logs.output))


class TestResolveFrameRate(unittest.TestCase):
    """The full resolution chain, and its refusal to default."""

    def _resolve(
        self,
        movie_path: Path,
        core: CoreMetadata,
        cli_frame_rate: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Resolve with the view derived from the same raw metadata."""
        return mu.resolve_frame_rate(
            mu.build_movie_qc_metadata(core),
            core,
            movie_path,
            cli_frame_rate,
        )

    def test_the_movie_rate_wins(self) -> None:
        """The rate that produced the movie is preferred."""
        with tempfile.TemporaryDirectory() as tmp:
            movie = fixtures.write_registered_movie(Path(tmp))
            rate, source = self._resolve(movie, acquisition_core())
        self.assertEqual(rate, fixtures.UPSTREAM_FRAME_RATE)
        self.assertEqual(source, mu.SOURCE_REGISTERED_MOVIE)

    def test_falls_back_to_the_acquisition_metadata(self) -> None:
        """A movie that records no rate drops to the acquisition file."""
        with tempfile.TemporaryDirectory() as tmp:
            movie = fixtures.write_registered_movie(Path(tmp), with_metadata=False)
            with self.assertLogs(mu.logger, level=logging.WARNING):
                rate, source = self._resolve(movie, acquisition_core())
        self.assertEqual(rate, fixtures.FRAME_RATE)
        self.assertEqual(source, mu.SOURCE_ACQUISITION)

    def test_falls_back_to_the_cli_override(self) -> None:
        """The operator's escape hatch is the last resort, not a default."""
        with tempfile.TemporaryDirectory() as tmp:
            movie = fixtures.write_registered_movie(Path(tmp), with_metadata=False)
            with self.assertLogs(mu.logger, level=logging.WARNING):
                rate, source = self._resolve(
                    movie, acquisition_core(frame_rate=None), cli_frame_rate=11.0
                )
        self.assertEqual(rate, 11.0)
        self.assertEqual(source, mu.SOURCE_CLI)

    def test_no_source_at_all_fails_loudly(self) -> None:
        """A wrong frame rate silently rescales output, so never default."""
        with tempfile.TemporaryDirectory() as tmp:
            movie = fixtures.write_registered_movie(Path(tmp), with_metadata=False)
            with self.assertLogs(mu.logger, level=logging.WARNING):
                with self.assertRaises(ValueError):
                    self._resolve(movie, acquisition_core(frame_rate=None))


class TestBuildCode(unittest.TestCase):
    """This library's identity fields on the Code block."""

    DISTRIBUTION = "aind-ophys-movie-qc-library"
    REPOSITORY = "https://github.com/AllenNeuralDynamics/aind-ophys-movie-qc-library"

    def test_identity_is_this_librarys_distribution(self) -> None:
        """Name, version and url come from this library, never a wrapper."""
        code = mu.build_code(parameters={})
        self.assertEqual(code.name, self.DISTRIBUTION)
        self.assertEqual(code.version, version(self.DISTRIBUTION))
        self.assertEqual(code.url, self.REPOSITORY)
        self.assertIsNone(code.commit_hash)


if __name__ == "__main__":
    unittest.main()
