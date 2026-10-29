"""Acquisition-metadata reads for registered-movie QC.

A thin layer over the shared ``aind_pophys_metadata`` library:
:class:`aind_pophys_metadata.CoreMetadata` locates the acquisition document
once and answers every field, so this module never sees a schema version. It
assembles the ``MovieQCMetadata`` view, reads the frame rate motion correction
recorded on the registered movie, and builds this step's ``Code`` block.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
from aind_data_schema.components.identifiers import Code
from aind_pophys_metadata import CoreMetadata
from aind_pophys_metadata import processing as shared_processing


logger = logging.getLogger("Ophys movie QC")

PROCESS_NAME = "aind-ophys-movie-qc"
DATA_PROCESS_NAME = "Movie QC"

# Motion correction's run arguments, stored as JSON on the registered movie.
MOVIE_METADATA_DATASET = "metadata"
MOVIE_FRAME_RATE_KEY = "frame_rate_hz"

# Labels recorded next to the resolved frame rate in processing.json.
SOURCE_REGISTERED_MOVIE = "registered movie metadata"
SOURCE_ACQUISITION = "acquisition metadata"
SOURCE_CLI = "--frame_rate"


@dataclass(frozen=True)
class MovieQCMetadata:
    """Minimal metadata registered-movie QC consumes.

    Frozen with immutable field types so the view stays hashable.

    ``frame_rate`` is the acquisition file's link in the frame-rate chain
    (see :func:`resolve_frame_rate`), so it is optional here; the run fails
    only when the whole chain yields nothing.

    ``fov_scale_factor`` reaches the raw metrics dump as provenance and no QC
    metric surfaces it, so a missing value must never fail a run. The two
    identifiers are logging context only and stay optional.
    """

    frame_rate: Optional[float] = None
    fov_scale_factor: Optional[float] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None


def read_movie_frame_rate(movie_path: Path) -> Optional[float]:
    """Read the frame rate motion correction recorded on its registered movie.

    Motion correction stores its run arguments, including ``frame_rate_hz``,
    as JSON on the movie's HDF5 ``metadata`` dataset. That is the rate that
    actually produced the movie (for mesoscope it is derived from the sync
    file and can differ from the nominal acquisition value), and the dataset
    has the same shape regardless of schema version.

    A movie without the dataset or the key is reported at WARNING, naming the
    file and what was missing, so the miss is never silent. The caller then
    continues down the resolution chain rather than defaulting.

    Parameters
    ----------
    movie_path : Path
        The registered-movie HDF5.

    Returns
    -------
    float or None
        The frame rate in Hz, or ``None`` when the movie does not record one.
    """
    with h5py.File(movie_path, "r") as handle:
        if MOVIE_METADATA_DATASET not in handle:
            logger.warning(
                "%s has no %s dataset; falling back to the acquisition metadata "
                "for the frame rate.",
                movie_path,
                MOVIE_METADATA_DATASET,
            )
            return None
        recorded = json.loads(handle[MOVIE_METADATA_DATASET][()])
    value = recorded.get(MOVIE_FRAME_RATE_KEY)
    if value is None:
        logger.warning(
            "%s %s dataset carries no %s; falling back to the acquisition "
            "metadata for the frame rate.",
            movie_path,
            MOVIE_METADATA_DATASET,
            MOVIE_FRAME_RATE_KEY,
        )
        return None
    return float(value)


def resolve_frame_rate(
    metadata: MovieQCMetadata,
    core: CoreMetadata,
    movie_path: Path,
    cli_frame_rate: Optional[float] = None,
) -> Tuple[float, str]:
    """Resolve the frame rate, in preference order, or fail loudly.

    Order: the rate motion correction recorded on the registered movie (see
    :func:`read_movie_frame_rate`), then the acquisition metadata, then the
    command line. The last two steps are the shared
    :meth:`aind_pophys_metadata.CoreMetadata.get_frame_rate` chain called with
    ``required=True``, so this step fails only after the CLI fallback is
    exhausted, with the fleet-wide error message. There is deliberately no
    default, because a wrong frame rate silently rescales the epilepsy
    event-width window and every per-second photon statistic.

    Parameters
    ----------
    metadata : MovieQCMetadata
        The acquisition-metadata view; its frame rate identifies which of the
        two shared-chain sources supplied the value.
    core : aind_pophys_metadata.CoreMetadata
        The located acquisition metadata the shared chain reads.
    movie_path : Path
        The registered-movie HDF5.
    cli_frame_rate : float, optional
        Explicit override from ``--frame_rate``.

    Returns
    -------
    tuple of (float, str)
        The frame rate in Hz and a label naming the source it came from.

    Raises
    ------
    ValueError
        If no source supplies a frame rate.
    """
    rate = read_movie_frame_rate(movie_path)
    if rate is not None:
        return rate, SOURCE_REGISTERED_MOVIE
    rate = core.get_frame_rate(cli_override=cli_frame_rate, required=True)
    source = SOURCE_ACQUISITION if metadata.frame_rate is not None else SOURCE_CLI
    return rate, source


def build_movie_qc_metadata(core: CoreMetadata) -> MovieQCMetadata:
    """Assemble the movie-QC view from the located core metadata.

    Nothing here is required: every field is either optional provenance or
    one link in the frame-rate chain that :func:`resolve_frame_rate` closes,
    so the frame rate is read with ``required=False``.

    Parameters
    ----------
    core : aind_pophys_metadata.CoreMetadata
        Acquisition metadata from :meth:`aind_pophys_metadata.CoreMetadata.load`.

    Returns
    -------
    MovieQCMetadata
        The minimal fields this step consumes.
    """
    return MovieQCMetadata(
        frame_rate=core.get_frame_rate(required=False),
        fov_scale_factor=core.get_um_per_pixel(),
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
    )


def build_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the v2 ``Code`` block for registered-movie QC.

    The name, version and repository url come from this library's installed
    distribution metadata.

    Parameters
    ----------
    parameters : dict
        Run *input* parameters recorded on the code block. Summary statistics
        about the outputs belong in ``DataProcess.output_parameters``.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.

    Returns
    -------
    Code
        The populated code block.
    """
    return shared_processing.build_code(
        __name__,
        parameters=parameters,
        input_data=input_data,
    )
