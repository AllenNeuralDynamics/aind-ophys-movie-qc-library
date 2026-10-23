"""quality_control.json generation for registered-movie QC.

Movie-QC metric content over the shared ``aind_pophys_metadata`` QC builders.
Metric-producing functions *return* metrics and :func:`collect_metrics`
accumulates them, so ``job.run`` writes exactly one ``quality_control.json``
per plane (the collector pattern).

Grouping is carried by every metric's ``evaluation`` tag, matching
``QualityControl.default_grouping = ["evaluation"]``, and its ``type`` tag
follows from the group: the photon and event detection statistics are
informational, every other group is operational QC. Statuses are
auto-computed from the measured values, so
``aind_pophys_metadata.quality_control.pending_qc_status`` (which hardcodes
PENDING) is not usable and ``QCStatus`` is built directly. Every metric seeds
a status: an empty ``status_history`` is rejected at construction.

Every metric name is prefixed with the plane id so a multiplane run's metrics
stay distinct.
"""

from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_pophys_metadata import quality_control as shared_qc

OPERATIONAL_TYPE = "Operational QC"
INFORMATIONAL_TYPE = "Informational"
EVALUATOR = "Automated"

EVALUATION_INTENSITY = "Intensity Drift"
EVALUATION_EPILEPSY = "Epilepsy"
EVALUATION_EVENTS = "Event detection statistics"
EVALUATION_PIXELS = "Pixel Saturation"
EVALUATION_PHOTONS = "Photon detection statistics"
EVALUATION_ZDRIFT = "Z-drift"

# The ``type`` tag of every ``evaluation`` group.
EVALUATION_TYPES = {
    EVALUATION_INTENSITY: OPERATIONAL_TYPE,
    EVALUATION_EPILEPSY: OPERATIONAL_TYPE,
    EVALUATION_EVENTS: INFORMATIONAL_TYPE,
    EVALUATION_PIXELS: OPERATIONAL_TYPE,
    EVALUATION_PHOTONS: INFORMATIONAL_TYPE,
    EVALUATION_ZDRIFT: OPERATIONAL_TYPE,
}

# Automatic status thresholds, named rather than inline.
INTENSITY_FAIL_PERCENT = 20
INTENSITY_PENDING_PERCENT = 10
SATURATED_PIXELS_FAIL = 1000
SATURATED_PIXELS_PENDING = 800
ZDRIFT_QC_THRESHOLD_UM = 10

# Keys of the ``references`` mapping ``job`` passes to :func:`collect_metrics`,
# each a figure path relative to the results root.
REF_INTENSITY_PLOT = "intensity_plot"
REF_INTENSITY_HIST = "intensity_hist"
REF_EPILEPSY = "epilepsy"
REF_COMBINED = "roi_and_photon"
REF_ZDRIFT = "zdrift"


def _auto_status(status: Status) -> QCStatus:
    """Return an automatically-derived status stamped in US/Pacific.

    Parameters
    ----------
    status : Status
        The derived pass/pending/fail status.

    Returns
    -------
    QCStatus
        The status entry seeded into a metric's ``status_history``.
    """
    return QCStatus(
        evaluator=EVALUATOR,
        status=status,
        timestamp=dt.now(shared_qc.SEATTLE_TZ).isoformat(),
    )


def _metric(
    name: str,
    evaluation: str,
    value: Any,
    description: str,
    status: Status,
    reference: Optional[str] = None,
) -> QCMetric:
    """Build one automatically-evaluated QCMetric.

    Parameters
    ----------
    name : str
        Metric name, already prefixed with the plane id.
    evaluation : str
        Value of the ``evaluation`` tag, naming the group this metric joins;
        a key of ``EVALUATION_TYPES``, which supplies the ``type`` tag.
    value : Any
        The measured value; ``QCMetric.value`` is typed ``Any``, so a scalar,
        string or dict all validate.
    description : str
        Reviewer-facing description of what was measured and how to read it.
    status : Status
        The automatically-derived status.
    reference : str or None, optional
        Figure path relative to the results root.

    Returns
    -------
    QCMetric
        The populated metric.
    """
    return QCMetric(
        name=name,
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": evaluation, "type": EVALUATION_TYPES[evaluation]},
        description=description,
        reference=reference,
        status_history=[_auto_status(status)],
        value=value,
    )


def intensity_status(intensity_change: float) -> Status:
    """Derive the intensity-stability status from the percent change.

    Parameters
    ----------
    intensity_change : float
        Percent change in mean pixel intensity from movie start to end.

    Returns
    -------
    Status
        FAIL at or above ``INTENSITY_FAIL_PERCENT`` absolute change, PENDING
        at or above ``INTENSITY_PENDING_PERCENT``, else PASS.
    """
    magnitude = abs(float(intensity_change))
    if magnitude >= INTENSITY_FAIL_PERCENT:
        return Status.FAIL
    if magnitude >= INTENSITY_PENDING_PERCENT:
        return Status.PENDING
    return Status.PASS


def saturation_status(saturated_pixels: int) -> Status:
    """Derive the pixel-distribution status from the saturated-pixel count.

    Parameters
    ----------
    saturated_pixels : int
        Number of pixels at or above the saturation threshold.

    Returns
    -------
    Status
        FAIL at or above ``SATURATED_PIXELS_FAIL`` pixels, PENDING at or
        above ``SATURATED_PIXELS_PENDING``, else PASS.
    """
    if saturated_pixels >= SATURATED_PIXELS_FAIL:
        return Status.FAIL
    if saturated_pixels >= SATURATED_PIXELS_PENDING:
        return Status.PENDING
    return Status.PASS


def build_intensity_metric(
    unique_id: str,
    metrics: dict,
    reference: Optional[str] = None,
) -> QCMetric:
    """Percent change in mean pixel intensity across the movie.

    Parameters
    ----------
    unique_id : str
        Plane identifier used to prefix the metric name.
    metrics : dict
        The raw metrics dict; reads ``percent_change_intensity``.
    reference : str or None, optional
        Path to the intensity-progression figure, relative to the results
        root.

    Returns
    -------
    QCMetric
        The intensity-stability metric.
    """
    intensity_change = float(metrics.get("percent_change_intensity", 0.0))
    return _metric(
        name=f"{unique_id} Intensity stability",
        evaluation=EVALUATION_INTENSITY,
        value=f"{intensity_change:.2f}%",
        description=(
            "Percent change in mean intensity between a window near the start and one "
            "near the end of the movie. "
            f"Pass below {INTENSITY_PENDING_PERCENT}%, Pending below {INTENSITY_FAIL_PERCENT}%, "
            f"Fail at {INTENSITY_FAIL_PERCENT}% or more. A large drop suggests photobleaching."
        ),
        status=intensity_status(intensity_change),
        reference=reference,
    )


def build_epilepsy_metric(
    unique_id: str,
    metrics: dict,
    reference: Optional[str] = None,
) -> QCMetric:
    """Fraction of detected events that look epileptiform.

    Parameters
    ----------
    unique_id : str
        Plane identifier used to prefix the metric name.
    metrics : dict
        The raw metrics dict; reads ``epilepsy_probability``.
    reference : str or None, optional
        Path to the epilepsy figure, relative to the results root.

    Returns
    -------
    QCMetric
        The epilepsy-probability metric.
    """
    probability = float(metrics.get("epilepsy_probability", 0.0))
    return _metric(
        name=f"{unique_id} Epilepsy probability",
        evaluation=EVALUATION_EPILEPSY,
        value=probability,
        description=(
            "Fraction of detected events that look epileptiform (prominence above 10, "
            "width 0.1 to 0.3 s). Pass if zero, otherwise Fail; on Fail, look for "
            "clusters of large, brief events in the event plot."
        ),
        status=Status.FAIL if probability > 0 else Status.PASS,
        reference=reference,
    )


def build_event_detection_metric(
    unique_id: str,
    metrics: dict,
    short_movie_mode: bool = False,
) -> QCMetric:
    """Signal-to-noise and, when computed, ROI d-prime statistics.

    The two d-prime entries are omitted in short-movie mode rather than
    reported as 0.0, so a reviewer can tell "not computed" from a real zero.

    Parameters
    ----------
    unique_id : str
        Plane identifier used to prefix the metric name.
    metrics : dict
        The raw metrics dict; reads the ``simple_snr_*`` and, unless
        ``short_movie_mode``, the ``*_rois_dprime`` entries.
    short_movie_mode : bool, optional
        Whether the movie was too short for the stability-window metrics.

    Returns
    -------
    QCMetric
        The event-detection statistics metric.
    """
    values = {
        "SNR Mean": float(metrics.get("simple_snr_mean", 0.0)),
        "SNR Median": float(metrics.get("simple_snr_med", 0.0)),
        "SNR Standard Deviation": float(metrics.get("simple_snr_std", 0.0)),
    }
    if not short_movie_mode:
        values["Median ROI D-prime"] = float(metrics.get("median_rois_dprime", 0.0))
        values["ROI D-prime Standard Deviation"] = float(metrics.get("std_rois_dprime", 0.0))
    return _metric(
        name=f"{unique_id} Event detection statistics",
        evaluation=EVALUATION_EVENTS,
        value=values,
        description=(
            "Frame signal-to-noise ratio and per-ROI d-prime (Wilt et al., 2013), "
            "summarized across frames and ROIs. Low values point to weak signal or poor "
            "event detectability. D-prime is not computed for very short movies."
        ),
        status=Status.PASS,
    )


def build_pixel_distribution_metric(
    unique_id: str,
    metrics: dict,
    reference: Optional[str] = None,
) -> QCMetric:
    """Saturated, low-intensity and percentile pixel statistics.

    Parameters
    ----------
    unique_id : str
        Plane identifier used to prefix the metric name.
    metrics : dict
        The raw metrics dict; reads the ``nb_*_pixels*`` and ``percentile_*``
        entries.
    reference : str or None, optional
        Path to the intensity-histogram figure, relative to the results root.

    Returns
    -------
    QCMetric
        The pixel-value-distribution metric.
    """
    saturated_pixels = int(metrics.get("nb_saturated_pixels", 0))
    return _metric(
        name=f"{unique_id} Pixel value distribution",
        evaluation=EVALUATION_PIXELS,
        value={
            "Saturated Pixels": saturated_pixels,
            "Low Intensity Pixels": int(metrics.get("nb_low_pixels", 0)),
            "Low Intensity Pixels Percentage": float(metrics.get("nb_low_pixels_perc", 0.0)),
            "Intensity Percentile Lower (5th)": float(metrics.get("percentile_lower", 0.0)),
            "Intensity Percentile Upper (95th)": float(metrics.get("percentile_upper", 0.0)),
        },
        description=(
            "Saturated and low-intensity pixel counts over the first 500 frames of the "
            "analysed block, with the 5th and 95th percentiles. Pass below "
            f"{SATURATED_PIXELS_PENDING} saturated pixels, Pending from "
            f"{SATURATED_PIXELS_PENDING} to {SATURATED_PIXELS_FAIL - 1}, Fail at "
            f"{SATURATED_PIXELS_FAIL} or more."
        ),
        status=saturation_status(saturated_pixels),
        reference=reference,
    )


def build_photon_metric(
    unique_id: str,
    metrics: dict,
    reference: Optional[str] = None,
) -> QCMetric:
    """ROI segmentation, photon gain and photon-flux statistics.

    Parameters
    ----------
    unique_id : str
        Plane identifier used to prefix the metric name.
    metrics : dict
        The raw metrics dict; reads the segmentation and photon entries.
    reference : str or None, optional
        Path to the combined segmentation + Poisson figure, relative to the
        results root, since the QC portal cannot resolve an absolute path.

    Returns
    -------
    QCMetric
        The photon-detection statistics metric.
    """
    return _metric(
        name=f"{unique_id} Photon detection statistics",
        evaluation=EVALUATION_PHOTONS,
        value={
            "Mean ROI Intensity": float(metrics.get("mean_rois_intensity", 0.0)),
            "ROI Segmentation Threshold": float(metrics.get("rois_threshold", 0.0)),
            "Median ROI Neuropil Sum": float(metrics.get("median_sum_rois_neuropil", 0.0)),
            "Photon Gain": float(metrics.get("photon_gain", 0.0)),
            "Photon Offset": float(metrics.get("photon_offset", 0.0)),
            "Background Noise": float(metrics.get("background_noise", 0.0)),
            "Photon Flux Median": float(metrics.get("photon_flux_median_per_pixel_per_frame", 0.0)),
            "Mean Photons per ROI per Frame": float(
                metrics.get("mean_photons_per_roi_per_frame", 0.0)
            ),
            "Mean Photons per ROI per Second": float(
                metrics.get("mean_photons_per_roi_per_s", 0.0)
            ),
            "Mean Photons per Neuropil per Second": float(
                metrics.get("mean_photons_per_neuropil_per_s", 0.0)
            ),
        },
        description=(
            "Photon gain and offset fitted from pixel variance against mean, with the "
            "resulting background noise and the photon flux of ROIs and neuropil. "
            "Check for outliers."
        ),
        status=Status.PASS,
        reference=reference,
    )


def zdrift_values(zdrift: dict) -> Dict[str, Any]:
    """Flatten the z-drift result into the reviewer-facing value dict.

    Parameters
    ----------
    zdrift : dict
        The z-drift result, including its ``local_zstack_parameters`` entry.
        The per-plane ``start_corr``/``end_corr`` arrays are deliberately
        excluded; they are long and live in the raw metrics dump.

    Returns
    -------
    dict
        The ten flattened z-drift values.
    """
    zstack = zdrift.get("local_zstack_parameters") or {}
    return {
        "z_drift_um": float(zdrift["z_drift_um"]),
        "start_frame": int(zdrift["start_frame"]),
        "end_frame": int(zdrift["end_frame"]),
        "z_drift_frame": int(zdrift["z_drift_frame"]),
        "start_frame_corr": round(float(zdrift["start_frame_corr"]), 3),
        "end_frame_corr": round(float(zdrift["end_frame_corr"]), 3),
        "nb_of_loops": int(zstack["nb_of_loops"]),
        "nb_of_planes": int(zstack["nb_of_planes"]),
        "z_spacing_um": round(float(zstack["z_spacing_um"]), 2),
        "total_z_distance": round(float(zstack["total_z_distance"]), 2),
    }


def build_zdrift_metric(
    unique_id: str,
    zdrift: dict,
    reference: Optional[str] = None,
) -> QCMetric:
    """Z-drift between the start and end of the movie.

    Parameters
    ----------
    unique_id : str
        Plane identifier used to prefix the metric name.
    zdrift : dict
        The z-drift result.
    reference : str or None, optional
        Path to the z-drift figure, relative to the results root.

    Returns
    -------
    QCMetric
        The z-drift metric.
    """
    values = zdrift_values(zdrift)
    status = Status.PASS if abs(values["z_drift_um"]) <= ZDRIFT_QC_THRESHOLD_UM else Status.FAIL
    return _metric(
        name=f"{unique_id} Z-drift Analysis",
        evaluation=EVALUATION_ZDRIFT,
        value=values,
        description=(
            "Depth drift over the session: the start and end of the registered movie are "
            "each matched to the local z-stack, and the drift is the depth difference "
            "between the best-matching planes. "
            f"Pass within {ZDRIFT_QC_THRESHOLD_UM} um, otherwise Fail."
        ),
        status=status,
        reference=reference,
    )


def collect_metrics(
    unique_id: str,
    metrics: dict,
    references: Dict[str, str],
    short_movie_mode: bool = False,
    zdrift: Optional[dict] = None,
) -> List[QCMetric]:
    """Build every QC metric for one plane, applying the run's gating.

    All gating lives here so ``job`` stays thin. In short-movie mode the
    intensity-stability and photon-detection metrics are not emitted at all:
    their inputs were never computed, so emitting them would report
    fabricated zeros.

    Parameters
    ----------
    unique_id : str
        Plane identifier used to prefix every metric name.
    metrics : dict
        The raw metrics dict accumulated during the run.
    references : dict
        Figure paths relative to the results root, keyed by the ``REF_*``
        constants. A missing key yields a metric with no reference.
    short_movie_mode : bool, optional
        Whether the movie was too short for the stability-window metrics.
    zdrift : dict or None, optional
        The z-drift result, ``None`` when no local z-stack was mounted. The
        z-drift metric is emitted only with a result; without a z-stack there
        is nothing for a reviewer to assess.

    Returns
    -------
    list of QCMetric
        The metrics to write into one ``quality_control.json``.
    """
    collected = [
        build_epilepsy_metric(unique_id, metrics, references.get(REF_EPILEPSY)),
        build_event_detection_metric(unique_id, metrics, short_movie_mode=short_movie_mode),
        build_pixel_distribution_metric(unique_id, metrics, references.get(REF_INTENSITY_HIST)),
    ]
    if not short_movie_mode:
        collected.append(
            build_intensity_metric(unique_id, metrics, references.get(REF_INTENSITY_PLOT))
        )
        collected.append(build_photon_metric(unique_id, metrics, references.get(REF_COMBINED)))
    if zdrift is not None:
        collected.append(build_zdrift_metric(unique_id, zdrift, references.get(REF_ZDRIFT)))
    return collected


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl,
    output_dir: Path,
) -> Path:
    """Serialize a QualityControl object to ``quality_control.json``.

    Parameters
    ----------
    quality_control : QualityControl
        The QC document to write.
    output_dir : Path
        Directory into which ``quality_control.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_qc.write_quality_control_json(quality_control, output_dir)
