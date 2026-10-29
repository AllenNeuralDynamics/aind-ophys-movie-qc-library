if __name__ == "__main__":  # pragma: nocover
    # Create an ArgumentParser object

    start_time = dt.now()

    # Parse command-line arguments
    args = parse_args()
    # General settings

    # name of the dataset in the hdf5 file
    dataset_name = "data"

    output_dir = Path(args.output_dir)
    h5_file = next(Path(args.input_dir).rglob("*registered.h5"))
    unique_id = "_".join(h5_file.name.split("_")[:-1])
    output_dir = make_output_directory(output_dir, unique_id)

    frame_rate = args.frame_rate

    if frame_rate == 0:
        processing_json_fp = next(h5_file.parent.glob("*data_process.json"))
        with open(processing_json_fp, "r") as j:
            data = json.load(j)
        frame_rate = data["parameters"]["movie_frame_rate_hz"]

    with h5py.File(h5_file, "r") as h5_pointer:
        data_pointer = h5_pointer[dataset_name]

        # in principle, this is done once to avoid loading the data multiple times
        # We generate a few smaller version of the big movie to use for metrics
        cropped_video = subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=1,
            crop=args.crop,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        full_length_binning = 500
        small_cropped_video = cropped_video[0:full_length_binning, :, :]
        full_length_cropped_video = subsample_and_crop_video(
            data_pointer=data_pointer,
            subsample=full_length_binning,
            crop=args.crop,
            start_frame=0,
            end_frame=-1,
        )

        shape = data_pointer.shape
        short_movie_mode = shape[0] < SHORT_MOVIE_MODE_THRESHOLD

        # We extract a few frames from the beginning and end of the movie
        # to use for stability metrics
        if not short_movie_mode:
            start_section = subsample_and_crop_video(
                data_pointer=data_pointer,
                subsample=1,
                crop=args.crop,
                start_frame=IGNORE_FRAMES,
                end_frame=NB_BORDER_FRAMES_TO_AVG + IGNORE_FRAMES,
            )
            end_section = subsample_and_crop_video(
                data_pointer=data_pointer,
                subsample=1,
                crop=args.crop,
                start_frame=-(IGNORE_FRAMES + NB_BORDER_FRAMES_TO_AVG + 1),
                end_frame=-IGNORE_FRAMES,
            )
        else:
            logging.warning(
                "Movie has %d frames (< SHORT_MOVIE_MODE_THRESHOLD=%d); "
                "skipping stability-window metrics (intensity drift, "
                "segmentation, photon flux per-ROI, d-prime) and their QC "
                "evaluations.",
                shape[0], SHORT_MOVIE_MODE_THRESHOLD,
            )
            start_section = None
            end_section = None

    metrics = {}
    metrics["crops"] = args.crop
    metrics["shape"] = shape
    if not short_movie_mode:
        metrics["percent_change_intensity"] = get_percent_change_intensity(
            start_section, end_section
        )
    metrics["epilepsy_probability"], fig_epilespy = get_and_plot_epilepsy_probability(
        cropped_video, frame_rate=frame_rate
    )

    # We get the input filename without the path
    base_file = os.path.basename(h5_file)
    # We remove extension
    base_file = os.path.splitext(base_file)[0]

    save_figure_to_storage(fig_epilespy, output_dir, base_file, "epilepsy_probability")

    # The following add the dictionary returned by
    # get_simple_snr_metrics to the metrics dictionary
    metrics.update(get_simple_snr_metrics(cropped_video))

    photon_gain_parameters = get_photon_gain_parameters(
        cropped_video, max_pixel_range=args.max_pixel_range
    )

    metrics.update(
        get_saturation_metrics(
            small_cropped_video,
            min_pixel_range=args.min_pixel_range,
            max_pixel_range=args.max_pixel_range,
        )
    )

    metrics.update(get_percentile_metrics(small_cropped_video, perc_min=5, perc_max=95))

    save_figure_to_storage(
        plot_projection_image(small_cropped_video),
        output_dir,
        base_file,
        "mean_projection_image",
    )

    metrics.update(photon_gain_parameters)

    if not short_movie_mode:
        save_figure_to_storage(
            plot_projection_image(start_section),
            output_dir,
            base_file,
            "start_projection_image",
        )

        save_figure_to_storage(
            plot_projection_image(end_section),
            output_dir,
            base_file,
            "end_projection_image",
        )

        rois_data, roi_figure = get_and_plot_basic_segmentation(start_section)

        save_figure_to_storage(
            roi_figure, output_dir, base_file, "basic_segmentation_image"
        )

        metrics.update(rois_data)

        # We use the photon gain and offset to convert segmented intensities to
        # photon flux
        metrics["all_rois_photons_per_rois_per_frame"] = (
            convert_intensity_into_photon_flux(
                metrics["sum_rois_intensity"],
                metrics["photon_offset"],
                metrics["photon_gain"],
            )
        )
        metrics["all_neuropils_photons_per_rois_per_frame"] = (
            convert_intensity_into_photon_flux(
                metrics["sum_rois_neuropil"],
                metrics["photon_offset"],
                metrics["photon_gain"],
            )
        )
        metrics["mean_photons_per_roi_per_frame"] = np.mean(
            metrics["all_rois_photons_per_rois_per_frame"]
        )
        metrics["std_photons_per_roi_per_frame"] = np.std(
            metrics["all_rois_photons_per_rois_per_frame"]
        )
        metrics["mean_photons_per_roi_per_s"] = frame_rate * np.mean(
            metrics["all_rois_photons_per_rois_per_frame"]
        )
        metrics["std_photons_per_roi_per_s"] = frame_rate * np.std(
            metrics["all_rois_photons_per_rois_per_frame"]
        )
        metrics["mean_photons_per_neuropil_per_s"] = frame_rate * np.mean(
            metrics["all_neuropils_photons_per_rois_per_frame"]
        )
        metrics["std_photons_per_neuropil_per_s"] = frame_rate * np.std(
            metrics["all_neuropils_photons_per_rois_per_frame"]
        )

        # Convert to dprime for spike detection

        # We first convert photons into photons per second
        all_rois_photons_per_second = (
            metrics["all_rois_photons_per_rois_per_frame"] * frame_rate
        )
        all_neuropils_photons_per_second = (
            metrics["all_neuropils_photons_per_rois_per_frame"] * frame_rate
        )

        metrics["all_rois_dprime"] = get_dprime_indicator(
            all_rois_photons_per_second,
            all_neuropils_photons_per_second,
            args.decay_time,
            args.dff_single_event_size,
        )

        metrics["median_rois_dprime"] = np.median(metrics["all_rois_dprime"])
        metrics["std_rois_dprime"] = np.std(metrics["all_rois_dprime"])

    save_figure_to_storage(
        plot_avg_intensity_progression(
            full_length_cropped_video, binning=full_length_binning
        ),
        output_dir,
        base_file,
        "physio_intensity_plot",
    )

    save_figure_to_storage(
        plot_intensity_histogram(
            cropped_video,
            min_range=args.min_pixel_range,
            max_range=args.max_pixel_range,
        ),
        output_dir,
        base_file,
        "physio_intensity_hist",
    )

    save_figure_to_storage(
        plot_poisson_curve(photon_gain_parameters),
        output_dir,
        base_file,
        "physio_poisson_plot",
    )

    # z-drift metrics
    session_json_path = next(Path(args.input_dir).rglob("session.json"))
    try:
        zstack_filepath = next(
            Path(args.input_dir).rglob(f"{unique_id}_z_stack_local.h5")
        )
        local_zstack = LocalZStack(
            zstack_filepath=zstack_filepath,
            physio_filepath=h5_file,
            session_json_path=session_json_path,
        )
        metrics["zdrift"], save_imgs = (
            local_zstack.get_z_drift()
        )  # TODO: expose parameters

        metrics["zdrift"]["local_zstack_parameters"] = local_zstack.meta

        # Make the image
        fig = plt.figure(figsize=(10, 15))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 0.5], figure=fig)
        axes = np.empty((3, 2), dtype=object)

        image_segments = ["start", "end"]
        image_types = ["image", "zstack_plane"]
        for yi, image_segment in enumerate(image_segments):
            for xi, image_type in enumerate(image_types):
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
                axes[yi, xi] = ax

        ax_corr = fig.add_subplot(gs[2, :])
        start_corr = metrics["zdrift"]["start_corr"]
        end_corr = metrics["zdrift"]["end_corr"]
        start_idx = metrics["zdrift"]["start_frame"]
        end_idx = metrics["zdrift"]["end_frame"]
        zdrift_um = metrics["zdrift"]["z_drift_um"]

        ax_corr.plot(start_corr, label="start_corr")
        ax_corr.plot(end_corr, label="end_corr")
        ax_corr.legend()
        ax_corr.axvline(start_idx, color="k", linestyle="--")
        ax_corr.axvline(end_idx, color="k", linestyle="--")
        ax_corr.set_title(
            f"Start frame {start_idx}, end frame {end_idx}, zdrift = {zdrift_um} um"
        )
        ax_corr.set_xlabel("Z plane")
        ax_corr.set_ylabel("Correlation")

        fig.tight_layout()

        # Save the image
        save_figure_to_storage(fig, output_dir, base_file, f"zdrift", dpi=300)

        # save z-stack files (both raw and registered)
        zstack_reg_save_filepath = (
            Path(args.output_dir) / f"{unique_id}/movie_qc/{zstack_filepath.stem}_reg.h5"
        )
        with h5py.File(zstack_reg_save_filepath, "w") as h:
            h.create_dataset("data", data=local_zstack.zstack)
        qc_z_drift = True
    except StopIteration:
        logging.warning("No local z-stack found, skipping z-drift metrics.")
        qc_z_drift = False

    # We remove stuff we don't need to save that would take space.
    # These come from photon_gain_parameters and always exist.
    metrics.pop("mean")
    metrics.pop("var")
    metrics.pop("all_pixels_photon_per_pixel_per_frame")
    # These only exist when stability-window metrics ran; pass a default so
    # the pop is a no-op (rather than KeyError) on the short-movie skip path.
    metrics.pop("all_rois_intensity", None)
    metrics.pop("all_rois_dprime", None)
    metrics.pop("all_rois_photons_per_rois_per_frame", None)
    metrics.pop("sum_rois_intensity", None)
    metrics.pop("sum_rois_neuropil", None)
    metrics.pop("all_neuropils_photons_per_rois_per_frame", None)

    # We save the metrics to a json file
    with open(os.path.join(output_dir, base_file + "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    write_qc_evaluation(
        output_dir,
        unique_id,
        metrics,
        z_drift=qc_z_drift,
        short_movie_mode=short_movie_mode,
    )
