"""Movie-QC computation and figure rendering.

These routines produce the values reviewers read in QC Portal, so their
numerics are held fixed. Coverage-omitted - they need a real movie to exercise.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from oasis.functions import deconvolve as oasis_deconvolve
from scipy import ndimage
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde
from skimage import filters, measure

logger = logging.getLogger("Ophys movie QC")

# Frames skipped at each end of the movie when extracting stability windows.
IGNORE_FRAMES = 300
# Frames averaged at each end, offset by IGNORE_FRAMES from the absolute ends
# of the movie.
NB_BORDER_FRAMES_TO_AVG = 200
# Minimum movie length (frames) required to run the full QC. Below this the
# run enters "short-movie mode" and skips the start/end stability window
# slicing along with everything it gates: intensity drift, basic ROI
# segmentation, photon flux per-ROI, d-prime, and the corresponding metrics.
SHORT_MOVIE_MODE_THRESHOLD = 2 * (IGNORE_FRAMES + NB_BORDER_FRAMES_TO_AVG) + 1

# Frames per bin when subsampling the whole movie for the intensity trace.
FULL_LENGTH_BINNING = 500

# Dataset name inside the registered-movie HDF5.
MOVIE_DATASET = "data"

REGISTERED_MOVIE_GLOB = "*registered.h5"


def get_and_plot_epilepsy_probability(
    cropped_video: np.ndarray,
    frame_rate: float,
    signal_threshold: float = 10.0,
    min_width: float = 0.1,
    max_width: float = 0.3,
) -> Tuple[float, Optional[plt.Figure]]:
    """Get the probability of epilepsy in this physio movie.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze
    frame_rate : float
        Frames per second of the signal
    signal_threshold : float
        A bound above which events are tagged as epileptic events
    min_width : float
        Lower bound on the widths within which events are tagged as epileptic
    max_width : float
        Upper bound on the widths within which events are tagged as epileptic

    Returns
    -------
    float
        The probability of epilepsy.
    matplotlib.figure.Figure or None
        A figure showing the epilepsy plot, or ``None`` when the deconvolved
        signal is flat and there is nothing to plot.

    Raises
    ------
    Exception
        If the point-density estimate for the event scatter does not converge.
    """
    avg_frames = np.mean(cropped_video, axis=(1, 2)).astype("double")

    perc_diff = -100 * (1 - avg_frames / avg_frames.mean())
    denoised_signal, spike_events, _, _, _ = oasis_deconvolve(perc_diff, penalty=1)

    if set(denoised_signal) == {0.0}:  # No signal
        # Two values, because every caller unpacks two.
        logger.warning("Deconvolved signal is flat; reporting zero epilepsy probability.")
        return 0.0, None

    prominence_data, widths = get_spike_prominence_and_width(
        denoised_signal, spike_events, frame_rate=frame_rate
    )
    nb_epileptic_events = len(
        prominence_data[
            (prominence_data > signal_threshold) & (widths > min_width) & (widths < max_width)
        ]
    )

    fig, axs = plt.subplots(2, 1)

    # Calculate the point density
    xy = np.vstack([widths, prominence_data])
    try:
        z = gaussian_kde(xy)(xy)
    except (LinAlgError, ValueError):
        raise Exception("Unable to form plot for epilepsy - check video for anomalies.")

    plt.sca(axs[0])
    plt.scatter(widths, prominence_data, c=z, s=10)
    plt.xlabel("Event Width (s)")
    plt.xlim(0, 1)
    plt.ylabel("Event Size")

    plt.sca(axs[1])
    plt.plot(perc_diff, "b-")
    plt.xlabel("Frame number")
    plt.ylabel("DFF")

    plt.tight_layout()
    return float(nb_epileptic_events) / len(prominence_data), fig


def get_spike_prominence_and_width(
    denoised_signal: np.ndarray,
    spike_events: np.ndarray,
    frame_rate: float,
    spike_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the local prominence and width of each spike in a calcium trace.

    Local prominence is a measure of the strength of the spike relative to
    other spikes nearest to it.

    Parameters
    ----------
    denoised_signal : np.ndarray
        The denoised fluorescence signal
    spike_events : np.ndarray
        Discretized deconvolved neural activity (spikes)
    frame_rate : float
        Frames per second of the signal, used to convert width to seconds
    spike_threshold : float
        A bound below which spikes will be thrown out

    Returns
    -------
    np.ndarray
        An array of prominences
    np.ndarray
        An array of widths
    """
    event_idxs = np.where(spike_events > spike_threshold)[0]
    filtered_signal = denoised_signal[event_idxs]

    _prominences = []
    _widths = []

    for event_idx in event_idxs:
        # Prominence calculation
        local_amplitude = denoised_signal[event_idx]
        local_diff = local_amplitude - filtered_signal
        larger_spike_idxs = event_idxs[local_diff < 0]

        # Emptiness, not truthiness, in this and the three guards below: index 0
        # is a real index, and `.any()` tests for a non-zero value instead.
        pre_spike_idxs = larger_spike_idxs[larger_spike_idxs < event_idx]
        if pre_spike_idxs.size == 0:
            pre_spike_idx = 0
        else:
            # Nearest spike larger than the local spike, on the left
            pre_spike_idx = np.max(pre_spike_idxs)

        # Minimum signal between prev large spike and current spike
        pre_spikes = denoised_signal[pre_spike_idx:event_idx]
        pre_min = np.min(pre_spikes)

        post_peak_idxs = larger_spike_idxs[larger_spike_idxs > event_idx]
        if post_peak_idxs.size == 0:
            post_peak_idx = len(denoised_signal)
        else:
            # Nearest spike larger than the local spike, on the right
            post_peak_idx = np.min(post_peak_idxs)

        # Minimum signal between post large spike and current spike
        post_spikes = denoised_signal[event_idx:post_peak_idx]
        post_min = np.min(post_spikes)

        local_min = np.max([pre_min, post_min])
        local_prominence = local_amplitude - local_min
        _prominences.append(local_prominence)

        # Width calculation
        pre_half_idxs = np.where(pre_spikes < local_amplitude - local_prominence / 2)[0]
        if pre_half_idxs.size == 0:
            pre_half_idx = len(pre_spikes)
        else:
            pre_half_idx = np.min(len(pre_spikes) - pre_half_idxs)

        post_half_idxs = np.where(post_spikes < local_amplitude - local_prominence / 2)[0]
        if post_half_idxs.size == 0:
            post_half_idx = len(post_spikes)
        else:
            post_half_idx = np.min(post_half_idxs)

        local_width = post_half_idx + pre_half_idx
        _widths.append(local_width)

    return np.array(_prominences), np.array(_widths) / frame_rate


def plot_intensity_histogram(
    cropped_video: np.ndarray,
    min_range: int,
    max_range: int,
    log_scale: bool = True,
) -> plt.Figure:
    """Obtain a plot of the intensity histogram.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to plot
    min_range : int
        The minimum pixel value to consider
    max_range : int
        The maximum pixel value to consider
    log_scale : bool
        Whether to plot the histogram on a log scale

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the intensity histogram
    """
    fig = plt.figure()
    cropped_video = cropped_video.flatten().copy()
    plt.hist(
        cropped_video,
        bins=100,
        range=(min_range, max_range),
        density=True,
        log=log_scale,
        histtype="bar",
        ec="black",
    )
    plt.xlabel("Pixel Value")
    plt.ylabel("pdf")
    return fig


def plot_projection_image(
    cropped_video: np.ndarray,
    min_range: int = 1,
    max_range: int = 99,
) -> plt.Figure:
    """Create a projection image of the video.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to plot
    min_range : int
        The minimum percentile to consider
    max_range : int
        The maximum percentile to consider

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the projection image
    """
    fig = plt.figure()
    image_project = np.mean(cropped_video, axis=0)
    list_pixel_limits = np.percentile(image_project.flatten(), [min_range, max_range])
    plt.imshow(image_project, cmap="gray", vmin=list_pixel_limits[0], vmax=list_pixel_limits[1])
    plt.axis("off")
    return fig


def get_and_plot_basic_segmentation(
    cropped_video: np.ndarray,
    min_object_size: int = 100,
    max_object_size: int = 300,
    sigma_segmentation: int = 30,
) -> Tuple[dict, plt.Figure]:
    """Get basic segmentation data and plot the results.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze
    min_object_size : int
        The minimum size of an object to consider
    max_object_size : int
        The maximum size of an object to consider
    sigma_segmentation : int
        The sigma value to use in the segmentation

    Returns
    -------
    dict
        A dictionary of ROI data
    matplotlib.figure.Figure
        A figure showing the basic segmentation
    """
    # We use a median to remove calcium events for F0
    background_image = np.median(cropped_video, axis=0)

    # We do a high pass to remove background fluctuations and flatten the image.
    sigma = sigma_segmentation
    neuropil = ndimage.gaussian_filter(background_image, sigma=sigma)
    neuropil_substracted = background_image - neuropil

    otsu_threshold = filters.threshold_otsu(neuropil_substracted)
    binary_image = neuropil_substracted > otsu_threshold

    figure = plt.figure()
    list_pixel_limits = np.percentile(neuropil_substracted.flatten(), [1, 99])
    plt.imshow(
        neuropil_substracted,
        cmap="gray",
        vmin=list_pixel_limits[0],
        vmax=list_pixel_limits[1],
    )
    plt.axis("off")

    # Label connected components and filter objects by size
    label_image = measure.label(binary_image)
    roi_intensities = []
    roi_sum = []
    neuropil_sum = []

    roi_data = {}
    roi_data["rois_threshold"] = otsu_threshold

    for region in measure.regionprops(label_image):
        if region.area > min_object_size and region.area < max_object_size:
            y1, x1, y2, x2 = region.bbox
            segmented_area_pixels = neuropil_substracted[y1:y2, x1:x2]
            segmented_area_pixels_neuropil = neuropil[y1:y2, x1:x2]

            local_mean = np.mean(segmented_area_pixels)
            roi_intensities.append(local_mean)
            roi_sum.append(np.sum(segmented_area_pixels))
            neuropil_sum.append(np.sum(segmented_area_pixels_neuropil))

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=1,
            )
            plt.gca().add_patch(rect)

    roi_data["nb_rois"] = len(roi_intensities)
    roi_data["mean_rois_intensity"] = np.mean(roi_intensities)
    roi_data["std_rois_intensity"] = np.std(roi_intensities)
    roi_data["all_rois_intensity"] = roi_intensities
    roi_data["sum_rois_intensity"] = roi_sum
    roi_data["median_sum_rois_intensity"] = np.median(roi_sum)
    roi_data["sum_rois_neuropil"] = neuropil_sum
    roi_data["median_sum_rois_neuropil"] = np.median(neuropil_sum)

    return roi_data, figure


def plot_poisson_curve(photon_gain_parameters: dict) -> plt.Figure:
    """Obtain a plot showing Poisson characteristics of the signal.

    Parameters
    ----------
    photon_gain_parameters : dict
        A dictionary of parameters related to the photon gain.

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the Poisson characteristics of the signal.
    """
    h, xedges, yedges = np.histogram2d(
        photon_gain_parameters["var"],
        photon_gain_parameters["mean"],
        bins=(200, 200),
    )
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    fig = plt.figure()
    plt.imshow(h, origin="lower", extent=extent, aspect="auto", cmap="Blues")
    plt.colorbar()
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.xlim(photon_gain_parameters["mean"].min(), photon_gain_parameters["mean"].max())
    plt.ylim(photon_gain_parameters["var"].min(), photon_gain_parameters["var"].max())

    mean_range = np.linspace(0, photon_gain_parameters["mean"].max(), num=200)
    background_noise_mean = (
        -photon_gain_parameters["photon_offset"] / photon_gain_parameters["photon_gain"]
    )
    plt.tight_layout()
    plt.plot(
        mean_range,
        photon_gain_parameters["photon_gain"] * (mean_range - background_noise_mean),
        "r",
    )

    return fig


def plot_avg_intensity_progression(cropped_video: np.ndarray, binning: int) -> plt.Figure:
    """Obtain a plot showing the average intensity over time.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to plot
    binning : int
        The amount of binning associated with cropped_video

    Returns
    -------
    matplotlib.figure.Figure
        A figure showing the average intensity progression.
    """
    fig = plt.figure()

    y_data = np.mean(cropped_video, axis=(1, 2))
    x_vals = list(binning * np.arange(0, y_data.shape[0]))
    plt.plot(x_vals, y_data, "b-")
    plt.xlabel("Frame #")
    plt.ylabel("Fluorescence Value")
    plt.title("physio_intensity_plot")
    plt.tight_layout()

    return fig


def get_saturation_metrics(
    cropped_video: np.ndarray,
    min_pixel_range: int,
    max_pixel_range: int,
) -> dict:
    """Return the number of pixels falling outside the usual dynamic range.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze
    min_pixel_range : int
        The minimum pixel value to consider
    max_pixel_range : int
        The maximum pixel value to consider

    Returns
    -------
    dict
        A dictionary of saturation metrics.
    """
    max_plane = cropped_video.max(axis=0).flatten()
    nb_saturated_pixels = int((max_plane >= max_pixel_range).sum())

    nb_undersat_pixels = (cropped_video.flatten() <= min_pixel_range).sum()
    nb_undersat_pixels = nb_undersat_pixels / float(cropped_video.shape[0])
    nb_undersat_pixels_perc = (
        100 * nb_undersat_pixels / float(cropped_video.shape[1] * cropped_video.shape[2])
    )

    return {
        "nb_saturated_pixels": nb_saturated_pixels,
        "nb_low_pixels": int(nb_undersat_pixels),
        "nb_low_pixels_perc": nb_undersat_pixels_perc,
    }


def get_percent_change_intensity(start_section: np.ndarray, end_section: np.ndarray) -> float:
    """Return the percent change in mean pixel intensity across the movie.

    Parameters
    ----------
    start_section : np.ndarray
        The beginning section of the movie to use.
    end_section : np.ndarray
        The end section of the movie to use.

    Returns
    -------
    float
        The change in mean pixel intensity, as a percentage of the start.
    """
    start_mean = np.mean(start_section, axis=(0, 1, 2))
    end_mean = np.mean(end_section, axis=(0, 1, 2))

    percent_change_intensity = 100 * (end_mean - start_mean) / start_mean
    return percent_change_intensity


def get_simple_snr_metrics(cropped_video: np.ndarray) -> dict:
    """Compute metrics related to the signal-to-noise level for the images.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze

    Returns
    -------
    dict
        A dictionary of metrics
    """
    # Simple SNR - https://en.wikipedia.org/wiki/Signal-to-noise_ratio_(imaging)
    simple_snr_array = cropped_video.mean(axis=(1, 2)) / cropped_video.std(axis=(1, 2))

    return {
        "simple_snr_mean": simple_snr_array.mean(),
        "simple_snr_med": np.median(simple_snr_array),
        "simple_snr_std": simple_snr_array.std(),
    }


def get_percentile_metrics(cropped_video: np.ndarray, perc_min: int, perc_max: int) -> dict:
    """Compute metrics related to the range of the signal.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze
    perc_min : int
        Minimum percentile value for filtering
    perc_max : int
        Maximum percentile value for filtering

    Returns
    -------
    dict
        A dictionary of percentile parameters.
    """
    return {
        "percentile_lower": np.percentile(cropped_video, perc_min),
        "percentile_upper": np.percentile(cropped_video, perc_max),
    }


def subsample_and_crop_video(
    data_pointer: Union[h5py.Dataset, np.ndarray],
    subsample: int,
    crop: Sequence[int],
    start_frame: int = 0,
    end_frame: int = -1,
) -> np.ndarray:
    """Subsample and crop a video. Also functions as a data_pointer load.

    Parameters
    ----------
    data_pointer : h5py.Dataset or np.ndarray
        The data pointer to the video data.
    subsample : int
        An integer specifying the amount of subsampling (1 = full movie).
    crop : sequence of int
        ``(px_y, px_x)`` specifying the number of pixels to remove per edge.
    start_frame : int, optional
        The index of the first desired frame, by default 0.
    end_frame : int, optional
        Exclusive end index of the frame range, by default -1.

    Returns
    -------
    np.ndarray
        The resultant array.

    Raises
    ------
    NotImplementedError
        If ``data_pointer`` is neither an HDF5 dataset nor an ndarray.
    """
    if not isinstance(data_pointer, (h5py.Dataset, np.ndarray)):
        raise NotImplementedError("Data pointer must be a h5py dataset or numpy array.")

    _shape = data_pointer.shape
    px_y_start, px_x_start = crop
    px_y_end = _shape[1] - px_y_start
    px_x_end = _shape[2] - px_x_start

    if start_frame == _shape[0] - 1 and (end_frame == -1 or end_frame == _shape[0]):
        cropped_video = data_pointer[
            start_frame::subsample, px_y_start:px_y_end, px_x_start:px_x_end
        ]
    else:
        cropped_video = data_pointer[
            start_frame:end_frame:subsample,
            px_y_start:px_y_end,
            px_x_start:px_x_end,
        ]

    return cropped_video


def convert_intensity_into_photon_flux(
    intensities: np.ndarray,
    photon_offset: float,
    photon_gain: float,
) -> np.ndarray:
    """Convert intensity values into photon flux.

    Parameters
    ----------
    intensities : np.ndarray
        The intensity values to convert.
    photon_offset : float
        The offset value for the photon conversion.
    photon_gain : float
        The gain value for the photon conversion.

    Returns
    -------
    np.ndarray
        The converted photon flux values.
    """
    background_noise_mean = -photon_offset / photon_gain
    photon_flux = (intensities - background_noise_mean) / photon_gain

    return photon_flux


def get_photon_gain_parameters(
    cropped_video: np.ndarray,
    max_pixel_range: int,
    perc_min: int = 3,
    perc_max: int = 90,
) -> dict:
    """Compute parameters related to the physio signal's photon gain.

    Parameters
    ----------
    cropped_video : np.ndarray
        The video to analyze.
    max_pixel_range : int
        The maximum pixel value to consider.
    perc_min : int
        Min value between 0-100 used in filtering based on percentile.
    perc_max : int
        Max value between 0-100 used in filtering based on percentile.

    Returns
    -------
    dict
        A dictionary of parameters related to the physio signal, used for both
        plots and metrics.

    Raises
    ------
    Exception
        If the variance-versus-mean least-squares fit does not converge.
    """
    # Remove saturated pixels
    idxs_not_saturated = np.where(cropped_video.max(axis=0).flatten() < max_pixel_range)

    _var = cropped_video.var(axis=0).flatten()[idxs_not_saturated]
    _mean = cropped_video.mean(axis=0).flatten()[idxs_not_saturated]

    # Remove pixels that deviate from Poisson stats
    _var_scale = np.percentile(_var, [perc_min, perc_max])
    _mean_scale = np.percentile(_mean, [perc_min, perc_max])

    # Remove outliers
    _var_bool = np.logical_and(_var > _var_scale[0], _var < _var_scale[1])
    _mean_bool = np.logical_and(_mean > _mean_scale[0], _mean < _mean_scale[1])
    _no_outliers = np.logical_and(_var_bool, _mean_bool)

    _var_filt = _var[_no_outliers]
    _mean_filt = _mean[_no_outliers]
    _mat = np.vstack([_mean_filt, np.ones(len(_mean_filt))]).T

    try:
        slope, offset = np.linalg.lstsq(_mat, _var_filt)[0]
    except LinAlgError:
        raise Exception("Unable to get photon metrics - check video for anomalies.")

    background_noise_mean = -offset / slope

    photon_per_pixel_per_frame = convert_intensity_into_photon_flux(
        cropped_video.flatten(), offset, slope
    )

    return {
        "var": _var_filt,
        "mean": _mean_filt,
        "photon_gain": slope,
        "photon_offset": offset,
        "photon_flux_median_per_pixel_per_frame": np.median(photon_per_pixel_per_frame),
        "all_pixels_photon_per_pixel_per_frame": photon_per_pixel_per_frame,
        "background_noise": background_noise_mean,
    }


def get_dprime_indicator(
    baseline_photon_flux: np.ndarray,
    neuropil_photon_flux: np.ndarray,
    decay_time: float,
    dff_single_event_size: float,
) -> np.ndarray:
    """Calculate the d-prime indicator for event detection.

    This original formula was introduced in Wilt et al, 2013. Default values
    of decay_time and dff_single_event_size are for GCaMP6f. We improved the
    Wilt et al formula to correct for the presence of non-responsive
    background neuropil, assuming a fixed DF/F given by the indicator. Note
    that if neuropil is null we end up with dff_single_event_size, as the
    correcting factor is 1.

    Parameters
    ----------
    baseline_photon_flux : np.ndarray
        The baseline photon flux values.
    neuropil_photon_flux : np.ndarray
        The neuropil photon flux values.
    decay_time : float
        The estimated 1/2 decay time of the event reporter (s).
    dff_single_event_size : float
        The estimated DF/F event size of the reporter for single events (au).

    Returns
    -------
    np.ndarray
        The d-prime indicator values.
    """
    corrected_dff_single_event_size = (
        dff_single_event_size * baseline_photon_flux / (baseline_photon_flux + neuropil_photon_flux)
    )
    d_prime = (
        np.sqrt((baseline_photon_flux + neuropil_photon_flux) * decay_time / 2)
        * corrected_dff_single_event_size
    )

    return d_prime


def save_figure_to_storage(
    figure: plt.Figure,
    storage_path: Path,
    root_filename: str,
    image_name: str,
    dpi: int = 120,
) -> Path:
    """Save a QC figure to the file system and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to save.
    storage_path : pathlib.Path
        The directory where the figure will be saved.
    root_filename : str
        The root name of the file.
    image_name : str
        The name of the image.
    dpi : int, optional
        The resolution in dots per inch, by default 120.

    Returns
    -------
    pathlib.Path
        The path written, so the caller can record it as a QC reference.
    """
    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    filepath = storage_path / f"{root_filename}_{image_name}.png"

    figure.savefig(filepath, dpi=dpi)
    plt.close(figure)
    return filepath


def make_output_directory(output_dir: Path, unique_id: str) -> Path:
    """Create this plane's ``<unique_id>/movie_qc`` output directory.

    Parameters
    ----------
    output_dir : pathlib.Path
        The results root.
    unique_id : str
        Plane identifier.

    Returns
    -------
    pathlib.Path
        The created directory.
    """
    plane_dir = Path(output_dir) / unique_id / "movie_qc"
    plane_dir.mkdir(parents=True, exist_ok=True)
    return plane_dir


def plot_zdrift(zdrift: dict, save_imgs: dict) -> plt.Figure:
    """Render the 3x2 z-drift figure.

    Parameters
    ----------
    zdrift : dict
        The z-drift result from ``LocalZStack.get_z_drift``.
    save_imgs : dict
        The start/end movie and z-stack plane images from the same call.

    Returns
    -------
    matplotlib.figure.Figure
        The z-drift figure.
    """
    fig = plt.figure(figsize=(10, 15))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.5], figure=fig)

    for yi, image_segment in enumerate(["start", "end"]):
        for xi, image_type in enumerate(["image", "zstack_plane"]):
            key = f"{image_segment}_{image_type}"
            ax = fig.add_subplot(gs[yi, xi])
            img = save_imgs[key]
            ax.imshow(
                img,
                cmap="gray",
                vmin=np.percentile(img.flatten(), 1),
                vmax=np.percentile(img.flatten(), 99),
            )
            ax.set_title(key)

    ax_corr = fig.add_subplot(gs[2, :])
    start_idx = zdrift["start_frame"]
    end_idx = zdrift["end_frame"]

    ax_corr.plot(zdrift["start_corr"], label="start_corr")
    ax_corr.plot(zdrift["end_corr"], label="end_corr")
    ax_corr.legend()
    ax_corr.axvline(start_idx, color="k", linestyle="--")
    ax_corr.axvline(end_idx, color="k", linestyle="--")
    ax_corr.set_title(
        f"Start frame {start_idx}, end frame {end_idx}, " f"zdrift = {zdrift['z_drift_um']} um"
    )
    ax_corr.set_xlabel("Z plane")
    ax_corr.set_ylabel("Correlation")

    fig.tight_layout()
    return fig
