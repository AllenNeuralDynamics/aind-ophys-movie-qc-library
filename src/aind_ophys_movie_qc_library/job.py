"""Registered-movie QC entry point for one plane."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from matplotlib.figure import Figure
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import CoreMetadata, find_one, relative_to_root
from aind_pophys_metadata.processing import collect_upstream_process_names
from aind_pophys_metadata.runner import StageContext, stage_guard
from log_schema.main import setup_logging

from . import analysis, qc
from .local_z_stack import LOCAL_ZSTACK_GLOB, LocalZStack
from .settings import MovieQCSettings
from .utils import metadata_utils as mu
from .utils.image_utils import build_combined_roi_photon_image

logger = logging.getLogger("Ophys movie QC")

PROCESS_NAME = mu.PROCESS_NAME

# Statistics that are computed but which no QC metric surfaces. They are
# recorded on the DataProcess so they are not lost when a reviewer only reads
# quality_control.json.
UNSURFACED_METRIC_KEYS = (
    "nb_rois",
    "std_rois_intensity",
    "median_sum_rois_intensity",
    "std_photons_per_roi_per_frame",
    "std_photons_per_roi_per_s",
    "std_photons_per_neuropil_per_s",
)

# Bulky arrays dropped before the raw metrics dump is written.
BULKY_METRIC_KEYS = (
    "mean",
    "var",
    "all_pixels_photon_per_pixel_per_frame",
    "all_rois_intensity",
    "all_rois_dprime",
    "all_rois_photons_per_rois_per_frame",
    "sum_rois_intensity",
    "sum_rois_neuropil",
    "all_neuropils_photons_per_rois_per_frame",
)


def _load_movie_windows(settings: MovieQCSettings, movie_path: Path) -> dict:
    """Read every slice of the movie the statistics need, once.

    Parameters
    ----------
    settings : MovieQCSettings
        Resolved settings; supplies the crop and the frame window.
    movie_path : Path
        The registered-movie HDF5.

    Returns
    -------
    dict
        ``cropped``, ``small_cropped``, ``full_length``, ``start_section``,
        ``end_section``, ``shape`` and ``short_movie_mode``.
    """
    with h5py.File(movie_path, "r") as handle:
        data_pointer = handle[analysis.MOVIE_DATASET]
        windows = {
            "cropped": analysis.subsample_and_crop_video(
                data_pointer=data_pointer,
                subsample=1,
                crop=settings.crop,
                start_frame=settings.start_frame,
                end_frame=settings.end_frame,
            ),
            "full_length": analysis.subsample_and_crop_video(
                data_pointer=data_pointer,
                subsample=analysis.FULL_LENGTH_BINNING,
                crop=settings.crop,
                start_frame=0,
                end_frame=-1,
            ),
        }
        windows["small_cropped"] = windows["cropped"][0 : analysis.FULL_LENGTH_BINNING, :, :]
        shape = data_pointer.shape
        windows["shape"] = tuple(int(dim) for dim in shape)
        short_movie_mode = shape[0] < analysis.SHORT_MOVIE_MODE_THRESHOLD
        windows["short_movie_mode"] = short_movie_mode

        if short_movie_mode:
            logger.warning(
                "Movie has %d frames (< SHORT_MOVIE_MODE_THRESHOLD=%d); skipping "
                "stability-window metrics (intensity drift, segmentation, photon flux "
                "per-ROI, d-prime) and the metrics that report them.",
                shape[0],
                analysis.SHORT_MOVIE_MODE_THRESHOLD,
            )
            windows["start_section"] = None
            windows["end_section"] = None
            return windows

        windows["start_section"] = analysis.subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=1,
            crop=settings.crop,
            start_frame=analysis.IGNORE_FRAMES,
            end_frame=analysis.NB_BORDER_FRAMES_TO_AVG + analysis.IGNORE_FRAMES,
        )
        windows["end_section"] = analysis.subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=1,
            crop=settings.crop,
            start_frame=-(analysis.IGNORE_FRAMES + analysis.NB_BORDER_FRAMES_TO_AVG + 1),
            end_frame=-analysis.IGNORE_FRAMES,
        )
        return windows


def _photon_and_dprime_metrics(
    settings: MovieQCSettings,
    metrics: dict,
    frame_rate: float,
) -> None:
    """Convert segmented intensities to photon flux and d-prime, in place.

    Parameters
    ----------
    settings : MovieQCSettings
        Resolved settings; supplies the reporter decay time and event size.
    metrics : dict
        The metrics dict; must already carry the segmentation sums and the
        photon gain/offset.
    frame_rate : float
        Movie frame rate in Hz.
    """
    metrics["all_rois_photons_per_rois_per_frame"] = analysis.convert_intensity_into_photon_flux(
        metrics["sum_rois_intensity"],
        metrics["photon_offset"],
        metrics["photon_gain"],
    )
    metrics["all_neuropils_photons_per_rois_per_frame"] = (
        analysis.convert_intensity_into_photon_flux(
            metrics["sum_rois_neuropil"],
            metrics["photon_offset"],
            metrics["photon_gain"],
        )
    )
    rois = metrics["all_rois_photons_per_rois_per_frame"]
    neuropils = metrics["all_neuropils_photons_per_rois_per_frame"]

    metrics["mean_photons_per_roi_per_frame"] = np.mean(rois)
    metrics["std_photons_per_roi_per_frame"] = np.std(rois)
    metrics["mean_photons_per_roi_per_s"] = frame_rate * np.mean(rois)
    metrics["std_photons_per_roi_per_s"] = frame_rate * np.std(rois)
    metrics["mean_photons_per_neuropil_per_s"] = frame_rate * np.mean(neuropils)
    metrics["std_photons_per_neuropil_per_s"] = frame_rate * np.std(neuropils)

    metrics["all_rois_dprime"] = analysis.get_dprime_indicator(
        rois * frame_rate,
        neuropils * frame_rate,
        settings.decay_time,
        settings.dff_single_event_size,
    )
    metrics["median_rois_dprime"] = np.median(metrics["all_rois_dprime"])
    metrics["std_rois_dprime"] = np.std(metrics["all_rois_dprime"])


def _compute_metrics(settings: MovieQCSettings, windows: dict) -> dict:
    """Compute every statistic that does not need a figure on disk.

    Parameters
    ----------
    settings : MovieQCSettings
        Resolved settings.
    windows : dict
        Movie slices from :func:`_load_movie_windows`.

    Returns
    -------
    dict
        The metrics dict, including the photon-gain parameters the Poisson
        figure plots.
    """
    metrics = {"crops": list(settings.crop), "shape": windows["shape"]}
    if not windows["short_movie_mode"]:
        metrics["percent_change_intensity"] = analysis.get_percent_change_intensity(
            windows["start_section"], windows["end_section"]
        )

    metrics.update(analysis.get_simple_snr_metrics(windows["cropped"]))
    photon_gain_parameters = analysis.get_photon_gain_parameters(
        windows["cropped"], max_pixel_range=settings.max_pixel_range
    )
    metrics.update(
        analysis.get_saturation_metrics(
            windows["small_cropped"],
            min_pixel_range=settings.min_pixel_range,
            max_pixel_range=settings.max_pixel_range,
        )
    )
    metrics.update(
        analysis.get_percentile_metrics(windows["small_cropped"], perc_min=5, perc_max=95)
    )
    metrics.update(photon_gain_parameters)
    return metrics


def _write_figures(
    settings: MovieQCSettings,
    windows: dict,
    metrics: dict,
    epilepsy_figure: Optional[Figure],
    context: dict,
) -> Tuple[dict, List[str]]:
    """Render and save every figure, returning results-relative references.

    Parameters
    ----------
    settings : MovieQCSettings
        Resolved settings.
    windows : dict
        Movie slices from :func:`_load_movie_windows`.
    metrics : dict
        The metrics dict; supplies the photon-gain parameters for the Poisson
        figure, and the segmentation, photon-flux and d-prime results are
        merged into it here.
    epilepsy_figure : matplotlib.figure.Figure or None
        The epilepsy figure, already produced with the probability.
    context : dict
        Run context; reads ``output_dir``, ``base_file``, ``unique_id``,
        ``results_root`` and ``frame_rate``.

    Returns
    -------
    tuple of (dict, list of str)
        Figure references keyed by the ``qc.REF_*`` constants, and the
        projection-figure paths, all relative to the results root.
    """
    output_dir = context["output_dir"]
    base_file = context["base_file"]
    results_root = context["results_root"]

    def _save(figure: Figure, image_name: str) -> Path:
        """Save a figure and return the path written."""
        return analysis.save_figure_to_storage(figure, output_dir, base_file, image_name)

    paths = {}
    projection_paths = [
        _save(analysis.plot_projection_image(windows["small_cropped"]), "mean_projection_image")
    ]
    if epilepsy_figure is not None:
        paths[qc.REF_EPILEPSY] = _save(epilepsy_figure, "epilepsy_probability")

    segmentation_path = None
    if not windows["short_movie_mode"]:
        for section, image_name in (
            ("start_section", "start_projection_image"),
            ("end_section", "end_projection_image"),
        ):
            projection_paths.append(
                _save(analysis.plot_projection_image(windows[section]), image_name)
            )
        rois_data, roi_figure = analysis.get_and_plot_basic_segmentation(windows["start_section"])
        segmentation_path = _save(roi_figure, "basic_segmentation_image")
        metrics.update(rois_data)
        _photon_and_dprime_metrics(settings, metrics, context["frame_rate"])

    paths[qc.REF_PROJECTIONS] = projection_paths[0]
    paths[qc.REF_INTENSITY_PLOT] = _save(
        analysis.plot_avg_intensity_progression(
            windows["full_length"], binning=analysis.FULL_LENGTH_BINNING
        ),
        "physio_intensity_plot",
    )
    paths[qc.REF_INTENSITY_HIST] = _save(
        analysis.plot_intensity_histogram(
            windows["cropped"],
            min_range=settings.min_pixel_range,
            max_range=settings.max_pixel_range,
        ),
        "physio_intensity_hist",
    )
    poisson_path = _save(analysis.plot_poisson_curve(metrics), "physio_poisson_plot")

    if segmentation_path is not None:
        paths[qc.REF_COMBINED] = build_combined_roi_photon_image(
            segmentation_path,
            poisson_path,
            output_dir / f"{context['unique_id']}_roi_and_photon_metrics.png",
        )

    references = {key: relative_to_root(results_root, path) for key, path in paths.items()}
    projections = [relative_to_root(results_root, path) for path in projection_paths]
    return references, projections


def _run_zdrift(
    settings: MovieQCSettings,
    metadata: mu.MovieQCMetadata,
    movie_path: Path,
    context: dict,
) -> Tuple[Optional[dict], Optional[str]]:
    """Measure z-drift when a matching local z-stack is mounted.

    Parameters
    ----------
    settings : MovieQCSettings
        Resolved settings; supplies the input directory.
    metadata : MovieQCMetadata
        Acquisition metadata; supplies the resolved FOV scale factor.
    movie_path : Path
        The registered-movie HDF5.
    context : dict
        Run context; reads ``output_dir``, ``base_file``, ``unique_id`` and
        ``results_root``.

    Returns
    -------
    tuple of (dict or None, str or None)
        The z-drift result and the z-drift figure path relative to the
        results root, both ``None`` when no z-stack is mounted.
    """
    unique_id = context["unique_id"]
    zstack_path = find_one(
        Path(settings.input_dir).resolve(),
        LOCAL_ZSTACK_GLOB.format(unique_id=unique_id),
        required=False,
    )
    if zstack_path is None:
        logger.warning("No local z-stack found for %s, skipping z-drift metrics.", unique_id)
        return None, None

    local_zstack = LocalZStack(
        zstack_filepath=zstack_path,
        physio_filepath=movie_path,
        fov_scale_factor=metadata.fov_scale_factor,
    )
    zdrift, save_imgs = local_zstack.get_z_drift()
    zdrift["local_zstack_parameters"] = local_zstack.meta

    reference = relative_to_root(
        context["results_root"],
        analysis.save_figure_to_storage(
            analysis.plot_zdrift(zdrift, save_imgs),
            context["output_dir"],
            context["base_file"],
            "zdrift",
            dpi=300,
        ),
    )

    registered_zstack = context["output_dir"] / f"{zstack_path.stem}_reg.h5"
    with h5py.File(registered_zstack, "w") as handle:
        handle.create_dataset("data", data=local_zstack.zstack)

    return zdrift, reference


def _write_metrics_json(context: dict, metrics: dict) -> None:
    """Write the raw metrics dump, minus the bulky per-pixel arrays.

    Parameters
    ----------
    context : dict
        Run context; reads ``output_dir`` and ``base_file``.
    metrics : dict
        The metrics dict; left unchanged.
    """
    dump = {key: value for key, value in metrics.items() if key not in BULKY_METRIC_KEYS}
    path = context["output_dir"] / f"{context['base_file']}_metrics.json"
    with open(path, "w") as handle:
        json.dump(dump, handle, indent=4, default=str)


def _unsurfaced_statistics(metrics: dict) -> dict:
    """Collect the statistics no QC metric surfaces, for ``output_parameters``.

    Parameters
    ----------
    metrics : dict
        The metrics dict.

    Returns
    -------
    dict
        The unsurfaced statistics present in this run.
    """
    return {key: float(metrics[key]) for key in UNSURFACED_METRIC_KEYS if key in metrics}


def _locate_plane(results_root: Path, input_dir: Path) -> dict:
    """Locate the registered movie and derive this task's plane context.

    Parameters
    ----------
    results_root : Path
        Resolved results root.
    input_dir : Path
        Resolved input directory, searched recursively for the movie.

    Returns
    -------
    dict
        Run context seeded with ``results_root``, ``movie_path``,
        ``unique_id``, ``base_file``, ``input_data`` and ``output_dir``.
    """
    movie_path = find_one(input_dir, analysis.REGISTERED_MOVIE_GLOB)
    unique_id = "_".join(movie_path.name.split("_")[:-1])
    return {
        "results_root": results_root,
        "movie_path": movie_path,
        "unique_id": unique_id,
        "base_file": movie_path.stem,
        "input_data": [relative_to_root(input_dir, movie_path)],
        "output_dir": analysis.make_output_directory(results_root, unique_id),
    }


def _run_movie_qc(
    settings: MovieQCSettings,
    metadata: mu.MovieQCMetadata,
    core: CoreMetadata,
    context: dict,
    stage: StageContext,
) -> None:
    """Run every QC computation for one plane and write all outputs."""
    movie_path = context["movie_path"]
    unique_id = context["unique_id"]

    frame_rate, frame_rate_source = mu.resolve_frame_rate(
        metadata, core, movie_path, settings.frame_rate
    )
    context["frame_rate"] = frame_rate
    logger.info("Resolved frame rate %s Hz from %s", frame_rate, frame_rate_source)
    stage.parameters.update({"frame_rate": frame_rate, "frame_rate_source": frame_rate_source})

    windows = _load_movie_windows(settings, movie_path)
    stage.output_parameters.update(
        {
            "short_movie_mode": windows["short_movie_mode"],
            "movie_shape": list(windows["shape"]),
        }
    )

    metrics = _compute_metrics(settings, windows)
    probability, epilepsy_figure = analysis.get_and_plot_epilepsy_probability(
        windows["cropped"], frame_rate=frame_rate
    )
    metrics["epilepsy_probability"] = probability

    references, projections = _write_figures(settings, windows, metrics, epilepsy_figure, context)

    zdrift, zdrift_reference = _run_zdrift(settings, metadata, movie_path, context)
    if zdrift is not None:
        metrics["zdrift"] = zdrift
        references[qc.REF_ZDRIFT] = zdrift_reference

    _write_metrics_json(context, metrics)
    stage.output_parameters.update(_unsurfaced_statistics(metrics))

    collected = qc.collect_metrics(
        unique_id=unique_id,
        metrics=metrics,
        references=references,
        projections=projections,
        short_movie_mode=windows["short_movie_mode"],
        zdrift=zdrift,
    )
    qc.write_quality_control_json(qc.build_quality_control(collected), context["output_dir"])


def run(settings: Optional[MovieQCSettings] = None) -> None:
    """Run registered-movie QC end to end.

    Parameters
    ----------
    settings : MovieQCSettings, optional
        Parsed settings; constructed from CLI/env when omitted.
    """
    settings = settings or MovieQCSettings()
    results_root = Path(settings.output_dir).resolve()
    input_dir = Path(settings.input_dir).resolve()

    core = CoreMetadata.load(input_dir)
    metadata = mu.build_movie_qc_metadata(core)
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )

    context = _locate_plane(results_root, input_dir)
    unique_id = context["unique_id"]

    with stage_guard(
        context["output_dir"],
        ProcessName.ANALYSIS,
        mu.build_code(
            parameters=settings.model_dump(mode="json"),
            input_data=context["input_data"],
        ),
        plane_id=unique_id,
        logger=logger,
        name=mu.DATA_PROCESS_NAME,
        output_path=relative_to_root(results_root, context["output_dir"]),
    ) as stage:
        stage.upstream_names = collect_upstream_process_names(
            input_dir, exclude=f"{unique_id}: {mu.DATA_PROCESS_NAME}"
        )
        _run_movie_qc(settings, metadata, core, context, stage)
