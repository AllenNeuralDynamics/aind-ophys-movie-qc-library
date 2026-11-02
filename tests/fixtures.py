"""Minimal metadata, movie and metrics fixtures shared by the tests.

Deliberately minimal: only the fields the getters actually read, so the tests
prove the library tolerates messy-but-sufficient metadata rather than
requiring a fully-validating document. The discriminator strings are resolved
from the schema classes, matching how the shared read layer works.

``[tool.coverage.run] source`` includes ``tests``, so every helper here is
measured and must stay exercised by a test.
"""

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from aind_data_schema.components.configs import ImagingConfig, PlanarImage
from aind_data_schema.components.coordinates import Scale
from aind_pophys_metadata.io import object_type_value

FRAME_RATE = 9.48
SUBJECT_ID = "783992"
DATASET_NAME = "multiplane-ophys_783992_2026-07-30_12-00-00"
FOV_SCALE_FACTOR = 0.78
UNIQUE_ID = "VISp_0"

# Frame rate motion correction recorded on its registered movie; deliberately
# different from FRAME_RATE so tests can tell the two sources apart.
UPSTREAM_FRAME_RATE = 10.75


def write_json(directory: Path, name: str, payload: dict) -> Path:
    """Write ``payload`` as JSON into ``directory/name``.

    Parameters
    ----------
    directory : Path
        Destination directory; created if needed.
    name : str
        File name.
    payload : dict
        JSON-serializable content.

    Returns
    -------
    Path
        The path written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload))
    return path


def v2_acquisition(
    frame_rate: Optional[float] = FRAME_RATE,
    subject_id: Optional[str] = SUBJECT_ID,
    fov_scale_factor: Optional[float] = FOV_SCALE_FACTOR,
) -> dict:
    """Return a minimal v2 ``acquisition.json`` dict.

    Parameters
    ----------
    frame_rate : float, optional
        Frame rate on the imaging config's sampling strategy; ``None`` omits
        the key.
    subject_id : str, optional
        Subject identifier; ``None`` omits the key.
    fov_scale_factor : float, optional
        Scale magnitude on the image-to-acquisition transform; ``None`` omits
        the transform entry.

    Returns
    -------
    dict
        The raw acquisition dict.
    """
    image = {
        "object_type": object_type_value(PlanarImage),
        "planes": [{"plane_index": 0, "targeted_structure": {"acronym": "VISp"}}],
        "image_to_acquisition_transform": [],
    }
    if fov_scale_factor is not None:
        image["image_to_acquisition_transform"].append(
            {
                "object_type": object_type_value(Scale),
                "scale": [fov_scale_factor, fov_scale_factor],
            }
        )
    config = {
        "object_type": object_type_value(ImagingConfig),
        "coordinate_system": {"axis_unit": "micrometer"},
        "images": [image],
    }
    if frame_rate is not None:
        config["sampling_strategy"] = {"frame_rate": frame_rate}
    acquisition = {
        "instrument_id": "MESO.1",
        "data_streams": [{"configurations": [config]}],
    }
    if subject_id is not None:
        acquisition["subject_id"] = subject_id
    return acquisition


def data_description(name: str = DATASET_NAME) -> dict:
    """Return a minimal ``data_description.json`` dict."""
    return {"name": name}


def write_registered_movie(
    directory: Path,
    frame_rate: Optional[float] = UPSTREAM_FRAME_RATE,
    with_metadata: bool = True,
) -> Path:
    """Write a tiny registered movie shaped like motion correction's output.

    Motion correction dumps its run arguments as UTF-8 JSON bytes on the
    movie's scalar ``metadata`` dataset; this reproduces that encoding with
    only the keys the reader needs.

    Parameters
    ----------
    directory : Path
        Destination directory; created if needed.
    frame_rate : float, optional
        Frame rate recorded as ``frame_rate_hz``; ``None`` omits the key.
    with_metadata : bool, optional
        Whether to write the ``metadata`` dataset at all.

    Returns
    -------
    Path
        The movie written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{UNIQUE_ID}_registered.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((2, 4, 4), dtype="int16"))
        if with_metadata:
            args = {"input_dir": "../data", "output_dir": "../results"}
            if frame_rate is not None:
                args["frame_rate_hz"] = frame_rate
            handle.create_dataset(name="metadata", data=json.dumps(args).encode("utf-8"))
    return path


def zdrift_result(z_drift_um: float = 4.0) -> dict:
    """Return a z-drift result shaped like ``LocalZStack.get_z_drift``.

    Typed as that method emits it: the two best-plane correlations are read out
    of a correlation array and so are numpy scalars, while the full correlation
    vectors go through ``ndarray.tolist()`` and arrive as plain lists.

    Parameters
    ----------
    z_drift_um : float, optional
        Drift in micrometres, which drives the metric's automatic status.

    Returns
    -------
    dict
        The z-drift result, including ``local_zstack_parameters``.
    """
    return {
        "start_frame": 20,
        "end_frame": 24,
        "z_drift_frame": 4,
        "z_drift_um": z_drift_um,
        "start_corr": [0.1, 0.9],
        "end_corr": [0.2, 0.8],
        "start_frame_corr": np.float64(0.912345),
        "end_frame_corr": np.float64(0.887654),
        "local_zstack_parameters": {
            "nb_of_loops": 10,
            "nb_of_planes": 40,
            "z_spacing_um": 1.0004,
            "total_z_distance": 39.0156,
            "data_shape": (400, 512, 512),
            "fov_scale_factor": FOV_SCALE_FACTOR,
        },
    }


def raw_metrics() -> dict:
    """Return a metrics dict shaped like a full-length run's.

    Typed as ``analysis`` emits it: every statistic derived from the movie is a
    numpy scalar, because it comes out of ``np.mean``/``np.median``/``np.std``,
    ``np.percentile``, ``threshold_otsu`` or ``np.linalg.lstsq``. The counts
    that ``analysis`` coerces with ``int()`` and the epilepsy probability it
    coerces with ``float()`` stay plain.
    """
    return {
        "crops": [30, 30],
        "shape": (2000, 512, 512),
        "percent_change_intensity": np.float64(-3.25),
        "epilepsy_probability": 0.0,
        "simple_snr_mean": np.float64(12.5),
        "simple_snr_med": np.float64(12.1),
        "simple_snr_std": np.float64(1.4),
        "median_rois_dprime": np.float64(2.2),
        "std_rois_dprime": np.float64(0.3),
        "nb_saturated_pixels": 12,
        "nb_low_pixels": 5,
        "nb_low_pixels_perc": np.float64(0.002),
        "percentile_lower": np.float64(100.0),
        "percentile_upper": np.float64(900.0),
        "rois_threshold": np.float64(42.0),
        "nb_rois": 17,
        "mean_rois_intensity": np.float64(55.0),
        "std_rois_intensity": np.float64(6.0),
        "median_sum_rois_intensity": np.float64(900.0),
        "median_sum_rois_neuropil": np.float64(700.0),
        "photon_gain": np.float64(2.5),
        "photon_offset": np.float64(-3.0),
        "background_noise": np.float64(1.2),
        "photon_flux_median_per_pixel_per_frame": np.float64(0.44),
        "mean_photons_per_roi_per_frame": np.float64(8.0),
        "std_photons_per_roi_per_frame": np.float64(1.0),
        "mean_photons_per_roi_per_s": np.float64(75.8),
        "std_photons_per_roi_per_s": np.float64(9.5),
        "mean_photons_per_neuropil_per_s": np.float64(60.2),
        "std_photons_per_neuropil_per_s": np.float64(7.7),
    }
