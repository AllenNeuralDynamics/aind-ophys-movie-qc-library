# -*- coding: utf-8 -*-
import os
import numpy as np
import h5py
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
import json
from pathlib import Path


class LocalZStack:
    """Local Z Stack

    Objects contain methods to generate QC metrics from the local z stack files
    captured at the Allen Institute.  These files are typically named '*_zstack_local.h5'
    or '*_zstack_local_dewarping.h5', depending on the rig.

    Args:
        zstack_filepath:  The path to the local z stack file
        physio_filepath:  The path to the physio movie to use in calculating z-drift
        session_json_path:  The path to the session json file containing metadata about the session

    Attributes:
        ophys_experiment:  A LimsOphysExperiment object associated with this video
        data_pointer:  A reference to the HDF5 file of interest
        meta: A LocalZStackMeta object corresponding to the metadata of this video
        _cache:  A cache that stores previously subsampled and cropped videos
        dataset_name:  A string used to refer to this QC set
        metrics:  A dict of parameters to save in the Mouse-Seeks database
        images:  A dict of image locations to save in the Mouse-Seeks database
        _physio_filepath:  The path to the physio movie to use in calculating z-drift
    """

    def __init__(self, zstack_filepath, physio_filepath, session_json_path):
        self.zstack_filepath = zstack_filepath
        self.physio_filepath = physio_filepath
        self.session_json_path = session_json_path

        self.meta = {}
        self.local_zstack_metadata()
        self.zstack = self.process_stack()

        self.images = {}
        self.dataset_name = "local_z_stack"

    def _calculate_qc_metrics(self):
        """Calculates QC metrics and saves to the parameter self.metrics

        Returns:
            None
        """
        z_drift_corr, _ = self.get_z_drift(gaussian_filter=True, metric="corr")
        self.metrics = {
            "shape": self.meta["data_shape"],
            "z_drift_corr_start_frame": z_drift_corr["start_frame"],
            "z_drift_corr_end_frame": z_drift_corr["end_frame"],
            "z_drift_corr_frame_diff": z_drift_corr["end_frame"]
            - z_drift_corr["start_frame"],
            "z_drift_corr_um_diff": (
                z_drift_corr["end_frame"] - z_drift_corr["start_frame"]
            )
            * self.meta["z_spacing_um"],
            "z_drift_start_frame_match_method": "algorithm",
            "z_drift_end_frame_match_method": "algorithm",
            "filepath": self.physio_filepath,
        }
        if "start_frame_corr" in z_drift_corr:
            self.metrics["start_frame_corr"] = z_drift_corr["start_frame_corr"]
            self.metrics["end_frame_corr"] = z_drift_corr["end_frame_corr"]
        elif "start_frame_ssim" in z_drift_corr:
            self.metrics["start_frame_ssim"] = z_drift_corr["start_frame_ssim"]
            self.metrics["end_frame_ssim"] = z_drift_corr["end_frame_ssim"]

        try:
            z_drift_ssim, _ = self.get_z_drift(
                gaussian_filter=True, sigma=5, metric="ssim"
            )
            self.metrics.update(
                {
                    "z_drift_ssim_start_frame": z_drift_ssim["start_frame"],
                    "z_drift_ssim_end_frame": z_drift_ssim["end_frame"],
                    "z_drift_ssim_frame_diff": z_drift_ssim["end_frame"]
                    - z_drift_ssim["start_frame"],
                    "z_drift_ssim_um_diff": (
                        z_drift_ssim["end_frame"] - z_drift_ssim["start_frame"]
                    )
                    * self.meta["z_spacing_um"],
                    "z_drift_start_frame_match_method": "algorithm",
                    "z_drift_end_frame_match_method": "algorithm",
                }
            )
        except ValueError:  # SSIM is optional
            logger.debug(
                "Unable to compute SSIM for session {}".format(
                    self.ophys_experiment.lims_id
                )
            )
            pass

    def process_stack(self):
        """Averages common z-planes across loops.
        #TODO: Need to correct for motion.

        Returns:
            An array of shape (self.meta['nb_of_planes'] - (ignore_top + ignore_bot), Y, X)
        """
        stack = []
        with h5py.File(self.zstack_filepath, "r") as f:
            local_z_stack = f["data"][()]
            for plane_ind in range(self.meta["nb_of_planes"]):
                single_plane_images = local_z_stack[
                    range(
                        plane_ind, self.meta["data_shape"][0], self.meta["nb_of_planes"]
                    ),
                    ...,
                ]
                stack.append(np.mean(single_plane_images, axis=0))
        stack = np.array(stack)

        return stack

    def _input_process(self, input_image, gaussian_filter=False, sigma=3, shift=(0, 0)):
        """Processes the local z stack and the input image.
         Z-stack is cropped, shifted, and blurred.  Input image is blurred.

        Args:
            input_image:  2D array to compare to the local z stack planes
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the 2D gaussian filter
            shift: A tuple denoting (px_y, px_x) the number of pixels to shift, default (0, 0)

        Returns:
            The processed stack (also cached) and input image as numpy arrays.
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
                zstack_shape[1] == input_image_shape[0]
                and zstack_shape[2] == input_image_shape[1]
            ), "Z stack shape {} does not match input image shape {}".format(
                zstack_shape, input_image_shape
            )
            y_range_input, x_range_input, y_range_zstack, x_range_zstack = (
                self._range_from_shift(shift)
            )
            z_stack = z_stack[
                :,
                y_range_zstack[0] : y_range_zstack[1],
                x_range_zstack[0] : x_range_zstack[1],
            ]
            input_image = input_image[
                y_range_input[0] : y_range_input[1], x_range_input[0] : x_range_input[1]
            ]

        return z_stack, input_image

    def _range_from_shift(self, shift):
        """Returns the range of pixels to crop from the images and z stack based on the shift.
        Args:
            shift: A tuple denoting (px_y, px_x) the number of pixels to shift.

        Returns:
            Tuples y_range_input, x_range_input, y_range_zstack, x_range_zstack
                each tuple contains the start and end pixel indices for cropping.
        """
        _, y, x = self.meta["data_shape"]

        y_range_input = (max(0, -shift[0]), y - max(0, shift[0]))
        x_range_input = (max(0, -shift[1]), x - max(0, shift[1]))
        y_range_zstack = (max(0, shift[0]), shift[0] if shift[0] < 0 else y)
        x_range_zstack = (max(0, shift[1]), shift[1] if shift[1] < 0 else x)
        return y_range_input, x_range_input, y_range_zstack, x_range_zstack

    def _get_shift(self, register, image):
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

    def plot_z_drift_scores(
        self,
        start_image,
        end_image,
        gaussian_filter=True,
        sigma=3,
        register=True,
        metric="corr",
    ):
        """Generate a score vs. depth plot for a given z-drift metric.

        Args:
            start_image, end_image:  2D arrays to compare to the local z stack planes
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the 2D gaussian filter
            shift: A tuple denoting (px_y, px_x) the number of pixels to shift, default (0, 0)
            metric:  Metric choice, either pearson correlation (corr) or ssim (ssim)

        Returns:
            A figure.
        """
        shift = self._get_shift(register=register, image=start_image)

        if metric == "corr":
            start_coefs = self.get_corr_z_planes(
                start_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
            )
            end_coefs = self.get_corr_z_planes(
                end_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
            )
        elif metric == "ssim":
            start_coefs = self.get_ssim_z_planes(
                start_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
            )
            end_coefs = self.get_ssim_z_planes(
                end_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
            )
        else:
            raise NotImplementedError("Unhandled z-drift metric type {}".format(metric))

        z_spacing = self.meta["z_spacing_um"]

        fig = plt.figure()
        plt.plot(
            np.arange(0, len(start_coefs)) * z_spacing,
            start_coefs,
            marker=".",
            label="Start {}".format(metric.upper()),
        )
        plt.plot(
            np.arange(0, len(end_coefs)) * z_spacing,
            end_coefs,
            marker=".",
            label="End {}".format(metric.upper()),
        )

        plt.axvline(start_coefs.argmax() * z_spacing, color="black")
        plt.axvline(end_coefs.argmax() * z_spacing, color="black")
        plt.legend()
        plt.ylabel("Score")
        plt.xlabel(r"Depth ($um$)")
        plt.tight_layout()
        return fig

    def get_corr_z_planes(
        self,
        input_image,
        gaussian_filter=True,
        sigma=3,
        shift=(0, 0),
        ignore_top=5,
        ignore_bot=5,
    ):
        """Compares input_image to each z-plane of the local z stack. Returns correlation coefficients.

        Args:
            input_image:  2D array to compare to the local z stack planes
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove from stack
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the 2D gaussian filter
            shift: A tuple denoting (px_y, px_x) the number of pixels to shift, default (0, 0)
            ignore_top:  Number of planes to ignore from the top
            ignore_bot:  Number of planes to ignore from the bottom

        Returns:
            An array of correlation coefficients.
        """
        z_stack, input_image = self._input_process(
            input_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
        )
        coeffs = [
            np.corrcoef(input_image.flatten(), z_plane.flatten())[0, 1]
            for z_plane in z_stack
        ]

        # Set ignore planes to 0 correlation
        coeffs[:ignore_top] = np.ones(ignore_top) * np.min(coeffs)
        coeffs[-ignore_bot:] = np.ones(ignore_bot) * np.min(coeffs)
        return np.array(coeffs)

    def get_ssim_z_planes(
        self,
        input_image,
        crop=(0, 0),
        gaussian_filter=False,
        sigma=5,
        shift=None,
        ignore_top=5,
        ignore_bot=5,
    ):
        """Compares input_image to each z-plane of the local z stack. Returns correlation coefficients.

        Args:
            input_image:  2D array to compare to the local z stack planes
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove from stack
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the 2D gaussian filter
            shift: A tuple denoting (0, px_y, px_x) the number of pixels to shift, or None
            ignore_top:  Number of planes to ignore from the top
            ignore_bot:  Number of planes to ignore from the bottom

        Returns:
            An array of correlation coefficients.
        """
        z_stack, input_image = self._input_process(
            input_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
        )
        ssim_values = [ssim(input_image, z_plane) for z_plane in z_stack]

        # Set ignore planes to 0 correlation
        ssim_values[:ignore_top] = np.ones(ignore_top) * np.min(ssim_values)
        ssim_values[-ignore_bot:] = np.ones(ignore_bot) * np.min(ssim_values)
        return np.array(ssim_values)

    def get_z_drift(
        self,
        nb_frames_to_avg=500,
        gaussian_filter=True,
        sigma=3,
        register=True,
        metric="corr",
    ):
        """Determine the amount of z-drift in the motion corrected physio movie..
        # Note: This code assumes stable motion during z-stack imaging.

        Args:
            nb_frames_to_avg:  Number of frames to average in the physio movie
            gaussian_filter:  A boolean denoting whether or not to apply a 2D gaussian filter
            sigma:  The width of the gaussian filter
            register: A boolean specifying whether or not to register start_image to projected zstack for shift
            save_images: A boolean specifying whether or not to save the resultant images from this calculation
            metric:  Metric choice, either pearson correlation (corr) or ssim (ssim)

        Returns:
            A dict of z-drift information.
            A dict of of images to save.
                All with shifts applied. (vmin and vmax to be calculated during imshow)
        """

        with h5py.File(self.physio_filepath, "r") as f:
            start_image = np.array(f["data"][:nb_frames_to_avg, ...]).mean(axis=0)
            end_image = np.array(f["data"][-(nb_frames_to_avg + 1) :, ...]).mean(axis=0)

        shift = self._get_shift(register=register, image=start_image)

        correlation_scores = {}
        if metric == "corr":
            start_corr = self.get_corr_z_planes(
                start_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
            )
            start_idx = int(np.argmax(start_corr))
            correlation_scores["start_frame_corr"] = start_corr[start_idx]
            end_corr = self.get_corr_z_planes(
                end_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
            )
            end_idx = int(np.argmax(end_corr))
            correlation_scores["end_frame_corr"] = end_corr[end_idx]
        elif metric == "ssim":
            start_corr = self.get_ssim_z_planes(
                start_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
            )
            start_idx = int(np.argmax(start_corr))
            correlation_scores["start_frame_ssim"] = start_corr[start_idx]
            end_corr = self.get_ssim_z_planes(
                end_image, gaussian_filter=gaussian_filter, sigma=sigma, shift=shift
            )
            end_idx = int(np.argmax(end_corr))
            correlation_scores["end_frame_ssim"] = end_corr[end_idx]
        else:
            raise NotImplementedError("Unhandled z-drift metric type {}".format(metric))
        z_drift_metrics = {
            "start_frame": start_idx,
            "end_frame": end_idx,
            "z_drift_frame": end_idx - start_idx,
            "z_drift_um": (end_idx - start_idx) * self.meta["z_spacing_um"],
            "start_corr": start_corr.tolist(),
            "end_corr": end_corr.tolist(),
        }
        z_drift_metrics.update(correlation_scores)

        y_range_input, x_range_input, y_range_zstack, x_range_zstack = (
            self._range_from_shift(shift)
        )

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

    def local_zstack_metadata(self):
        """Get scanimage metadata and ROI groups from a local z-stack
        and save to self.meta

        """
        zstack_path = Path(self.zstack_filepath)
        with h5py.File(zstack_path, "r") as f:
            if "scanimage_metadata" not in f:
                raise ValueError("scanimage_metadata not found in the h5 file")
            si = f["scanimage_metadata"][()]
        si = si.decode()
        si = json.loads(si)
        si_metadata = si[0]
        roi_groups = si[1]

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
        self.meta["total_z_distance"] = (self.meta["nb_of_planes"] - 1) * self.meta[
            "z_spacing_um"
        ]
        self.meta["data_shape"] = local_zstack_shape

        with open(self.session_json_path, "r") as f:
            session_json = json.load(f)
        xy_scale = float(
            session_json["data_streams"][0]["ophys_fovs"][0]["fov_scale_factor"]
        )
        if (
            session_json["data_streams"][0]["ophys_fovs"][0]["fov_scale_factor_unit"]
            == "um/pixel"
        ):
            self.meta["fov_scale_factor"] = xy_scale
        else:
            raise NotImplementedError(
                "Unhandled fov scale factor unit {}".format(
                    session_json["data_streams"][0]["ophys_fovs"][0][
                        "fov_scale_factor_unit"
                    ]
                )
            )

        if (
            self.meta["nb_of_loops"] * self.meta["nb_of_planes"]
            != self.meta["data_shape"][0]
        ):
            raise Exception("Number of frames in local z stack different from metadata")
