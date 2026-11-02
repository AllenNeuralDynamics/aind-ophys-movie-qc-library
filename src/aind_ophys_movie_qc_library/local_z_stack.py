"""Local z-stack registration and z-drift measurement.

The constructor takes an already-resolved ``fov_scale_factor`` rather than
reading acquisition metadata itself.

Coverage-omitted: every method needs a real z-stack HDF5.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation

logger = logging.getLogger("Ophys movie QC")

LOCAL_ZSTACK_GLOB = "{unique_id}_z_stack_local.h5"


class LocalZStack:
    """Local z-stack QC metrics.

    Generates QC metrics from the local z-stack files captured at the Allen
    Institute, named ``<unique_id>_z_stack_local.h5`` (see
    :data:`LOCAL_ZSTACK_GLOB`).

    Parameters
    ----------
    zstack_filepath : path-like
        The path to the local z-stack file.
    physio_filepath : path-like
        The path to the physio movie used in calculating z-drift.
    fov_scale_factor : float, optional
        Micrometres per pixel, already resolved from the acquisition
        metadata. Recorded in ``meta`` as provenance; ``None`` when the
        metadata does not carry it.

    Attributes
    ----------
    meta : dict
        Metadata read from the z-stack's ScanImage header.
    zstack : numpy.ndarray
        Loop-averaged z-stack of shape ``(nb_of_planes, Y, X)``.
    """

    def __init__(
        self,
        zstack_filepath: Union[str, Path],
        physio_filepath: Union[str, Path],
        fov_scale_factor: Optional[float] = None,
    ) -> None:
        """Read the z-stack metadata and build the loop-averaged stack."""
        self.zstack_filepath = zstack_filepath
        self.physio_filepath = physio_filepath
        self.fov_scale_factor = fov_scale_factor

        self.meta = {}
        self.local_zstack_metadata()
        self.zstack = self.process_stack()

    def process_stack(self) -> np.ndarray:
        """Average common z-planes across loops.

        Returns
        -------
        numpy.ndarray
            An array of shape ``(self.meta['nb_of_planes'], Y, X)``.
        """
        stack = []
        with h5py.File(self.zstack_filepath, "r") as f:
            local_z_stack = f["data"][()]
            # TODO: correct for motion between loops before averaging.
            for plane_ind in range(self.meta["nb_of_planes"]):
                single_plane_images = local_z_stack[
                    range(
                        plane_ind,
                        self.meta["data_shape"][0],
                        self.meta["nb_of_planes"],
                    ),
                    ...,
                ]
                stack.append(np.mean(single_plane_images, axis=0))
        stack = np.array(stack)

        return stack

    def _input_process(
        self,
        input_image: np.ndarray,
        gaussian_filter: bool = False,
        sigma: int = 3,
        shift: Optional[Tuple[int, int]] = (0, 0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process the local z-stack and the input image.

        The z-stack is cropped, shifted and blurred; the input image is
        blurred.

        Parameters
        ----------
        input_image : numpy.ndarray
            2D array to compare to the local z-stack planes.
        gaussian_filter : bool
            Whether to apply a 2D gaussian filter.
        sigma : int
            The width of the 2D gaussian filter.
        shift : tuple
            ``(px_y, px_x)`` number of pixels to shift.

        Returns
        -------
        numpy.ndarray
            The processed stack.
        numpy.ndarray
            The processed input image.
        """
        z_stack = self.zstack.copy()

        # Filter z stack
        if gaussian_filter:
            z_stack = scipy.ndimage.gaussian_filter(z_stack, (0, sigma, sigma))
            input_image = scipy.ndimage.gaussian_filter(input_image, sigma=sigma)

        # Crop z stack and the image based on shift
        if shift != (0, 0):
            zstack_shape = z_stack.shape
            input_image_shape = input_image.shape
            assert (
                zstack_shape[1] == input_image_shape[0] and zstack_shape[2] == input_image_shape[1]
            ), "Z stack shape {} does not match input image shape {}".format(
                zstack_shape, input_image_shape
            )
            (
                y_range_input,
                x_range_input,
                y_range_zstack,
                x_range_zstack,
            ) = self._range_from_shift(shift)
            z_stack = z_stack[
                :,
                y_range_zstack[0] : y_range_zstack[1],
                x_range_zstack[0] : x_range_zstack[1],
            ]
            input_image = input_image[
                y_range_input[0] : y_range_input[1],
                x_range_input[0] : x_range_input[1],
            ]

        return z_stack, input_image

    def _range_from_shift(
        self, shift: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Return the pixel ranges to crop from the images and the z-stack.

        Parameters
        ----------
        shift : tuple
            ``(px_y, px_x)`` number of pixels to shift.

        Returns
        -------
        tuple
            ``y_range_input``, ``x_range_input``, ``y_range_zstack``,
            ``x_range_zstack``, each a ``(start, end)`` pixel index pair.
        """
        _, y, x = self.meta["data_shape"]

        y_range_input = (max(0, -shift[0]), y - max(0, shift[0]))
        x_range_input = (max(0, -shift[1]), x - max(0, shift[1]))
        y_range_zstack = (max(0, shift[0]), shift[0] if shift[0] < 0 else y)
        x_range_zstack = (max(0, shift[1]), shift[1] if shift[1] < 0 else x)
        return y_range_input, x_range_input, y_range_zstack, x_range_zstack

    def _get_shift(self, register: bool, image: np.ndarray) -> Tuple[int, int]:
        """Return the registration shift between an image and the z-stack.

        Parameters
        ----------
        register : bool
            Whether to register at all; ``False`` yields no shift.
        image : numpy.ndarray
            The image to register against the z-projected stack.

        Returns
        -------
        tuple
            ``(px_y, px_x)`` integer shift.
        """
        if register:
            # Register start image to z projected stack
            if hasattr(self, "register_shift"):
                shift = self.register_shift
            else:
                stack = self.process_stack()
                stack_projection = np.max(stack, axis=0)
                shift_xy, _, _ = phase_cross_correlation(stack_projection, image)
                shift = (int(shift_xy[0]), int(shift_xy[1]))
                self.register_shift = shift
        else:
            shift = (0, 0)
        return shift

    def get_corr_z_planes(
        self,
        input_image: np.ndarray,
        gaussian_filter: bool = True,
        sigma: int = 3,
        shift: Optional[Tuple[int, int]] = (0, 0),
        ignore_top: int = 5,
        ignore_bot: int = 5,
    ) -> np.ndarray:
        """Correlate ``input_image`` against each z-plane of the local stack.

        Parameters
        ----------
        input_image : numpy.ndarray
            2D array to compare to the local z-stack planes.
        gaussian_filter : bool
            Whether to apply a 2D gaussian filter.
        sigma : int
            The width of the 2D gaussian filter.
        shift : tuple
            ``(px_y, px_x)`` number of pixels to shift.
        ignore_top : int
            Number of planes to ignore from the top.
        ignore_bot : int
            Number of planes to ignore from the bottom.

        Returns
        -------
        numpy.ndarray
            An array of correlation coefficients.
        """
        z_stack, input_image = self._input_process(
            input_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
        )
        coeffs = [
            np.corrcoef(input_image.flatten(), z_plane.flatten())[0, 1] for z_plane in z_stack
        ]

        # Set ignore planes to the minimum correlation
        coeffs[:ignore_top] = np.ones(ignore_top) * np.min(coeffs)
        coeffs[-ignore_bot:] = np.ones(ignore_bot) * np.min(coeffs)
        return np.array(coeffs)

    def get_ssim_z_planes(
        self,
        input_image: np.ndarray,
        gaussian_filter: bool = False,
        sigma: int = 5,
        shift: Optional[Tuple[int, int]] = None,
        ignore_top: int = 5,
        ignore_bot: int = 5,
    ) -> np.ndarray:
        """Compare ``input_image`` to each z-plane of the stack, using SSIM.

        Parameters
        ----------
        input_image : numpy.ndarray
            2D array to compare to the local z-stack planes.
        gaussian_filter : bool
            Whether to apply a 2D gaussian filter.
        sigma : int
            The width of the 2D gaussian filter.
        shift : tuple or None
            ``(px_y, px_x)`` number of pixels to shift, or ``None``.
        ignore_top : int
            Number of planes to ignore from the top.
        ignore_bot : int
            Number of planes to ignore from the bottom.

        Returns
        -------
        numpy.ndarray
            An array of SSIM values.
        """
        z_stack, input_image = self._input_process(
            input_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
        )
        ssim_values = [ssim(input_image, z_plane) for z_plane in z_stack]

        # Set ignore planes to the minimum SSIM
        ssim_values[:ignore_top] = np.ones(ignore_top) * np.min(ssim_values)
        ssim_values[-ignore_bot:] = np.ones(ignore_bot) * np.min(ssim_values)
        return np.array(ssim_values)

    def get_z_drift(
        self,
        nb_frames_to_avg: int = 500,
        gaussian_filter: bool = True,
        sigma: int = 3,
        register: bool = True,
        metric: str = "corr",
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Determine the amount of z-drift in the motion-corrected movie.

        Assumes stable motion during z-stack imaging.

        Parameters
        ----------
        nb_frames_to_avg : int
            Number of frames to average at each end of the physio movie.
        gaussian_filter : bool
            Whether to apply a 2D gaussian filter.
        sigma : int
            The width of the gaussian filter.
        register : bool
            Whether to register the start image to the projected z-stack.
        metric : str
            Either ``"corr"`` (Pearson correlation) or ``"ssim"``.

        Returns
        -------
        dict
            Z-drift information.
        dict
            Images to save, with shifts already applied.

        Raises
        ------
        NotImplementedError
            If ``metric`` is not one of the supported values.
        """
        with h5py.File(self.physio_filepath, "r") as f:
            start_image = np.array(f["data"][:nb_frames_to_avg, ...]).mean(axis=0)
            end_image = np.array(f["data"][-(nb_frames_to_avg + 1) :, ...]).mean(axis=0)

        shift = self._get_shift(register=register, image=start_image)

        if metric == "corr":
            score_z_planes = self.get_corr_z_planes
        elif metric == "ssim":
            score_z_planes = self.get_ssim_z_planes
        else:
            raise NotImplementedError("Unhandled z-drift metric type {}".format(metric))

        start_corr = score_z_planes(
            start_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
        )
        start_idx = int(np.argmax(start_corr))
        end_corr = score_z_planes(
            end_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
        )
        end_idx = int(np.argmax(end_corr))
        correlation_scores = {
            f"start_frame_{metric}": start_corr[start_idx],
            f"end_frame_{metric}": end_corr[end_idx],
        }
        z_drift_metrics = {
            "start_frame": start_idx,
            "end_frame": end_idx,
            "z_drift_frame": end_idx - start_idx,
            "z_drift_um": (end_idx - start_idx) * self.meta["z_spacing_um"],
            "start_corr": start_corr.tolist(),
            "end_corr": end_corr.tolist(),
        }
        z_drift_metrics.update(correlation_scores)

        (
            y_range_input,
            x_range_input,
            y_range_zstack,
            x_range_zstack,
        ) = self._range_from_shift(shift)

        save_imgs = {}
        save_imgs["start_image"] = start_image[
            y_range_input[0] : y_range_input[1], x_range_input[0] : x_range_input[1]
        ]
        save_imgs["end_image"] = end_image[
            y_range_input[0] : y_range_input[1], x_range_input[0] : x_range_input[1]
        ]
        save_imgs["start_zstack_plane"] = self.zstack[
            start_idx,
            y_range_zstack[0] : y_range_zstack[1],
            x_range_zstack[0] : x_range_zstack[1],
        ]
        save_imgs["end_zstack_plane"] = self.zstack[
            end_idx,
            y_range_zstack[0] : y_range_zstack[1],
            x_range_zstack[0] : x_range_zstack[1],
        ]

        return z_drift_metrics, save_imgs

    def local_zstack_metadata(self) -> None:
        """Read the z-stack's ScanImage metadata into ``self.meta``.

        Raises
        ------
        ValueError
            If the HDF5 file has no ``scanimage_metadata`` entry, or if
            ``data`` is not a dataset.
        Exception
            If the frame count disagrees with the ScanImage metadata.
        """
        zstack_path = Path(self.zstack_filepath)
        with h5py.File(zstack_path, "r") as f:
            if "scanimage_metadata" not in f:
                raise ValueError("scanimage_metadata not found in the h5 file")
            si = f["scanimage_metadata"][()]
        si = si.decode()
        si = json.loads(si)
        si_metadata = si[0]

        nb_of_loops = int(si_metadata["SI.hStackManager.actualNumVolumes"])
        nb_of_planes = int(si_metadata["SI.hStackManager.actualNumSlices"])
        z_spacing_um = float(si_metadata["SI.hStackManager.actualStackZStepSize"])
        with h5py.File(zstack_path, "r") as f:
            if isinstance(f["data"], h5py.Dataset):
                local_zstack_shape = f["data"].shape
            else:
                raise ValueError("'data' is not a dataset in the HDF5 file")

        self.meta["nb_of_loops"] = nb_of_loops
        self.meta["nb_of_planes"] = nb_of_planes
        self.meta["z_spacing_um"] = z_spacing_um
        self.meta["total_z_distance"] = (nb_of_planes - 1) * z_spacing_um
        self.meta["data_shape"] = local_zstack_shape
        self.meta["fov_scale_factor"] = self.fov_scale_factor
        if self.fov_scale_factor is None:
            logger.warning(
                "No fov_scale_factor resolved from the acquisition metadata; "
                "recording it as null in the z-drift provenance."
            )

        if nb_of_loops * nb_of_planes != local_zstack_shape[0]:
            raise Exception("Number of frames in local z stack different from metadata")
