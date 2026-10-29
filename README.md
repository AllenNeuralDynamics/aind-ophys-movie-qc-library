# aind-ophys-movie-qc-library

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Quality control for motion-corrected (registered) ophys movies.

This is the backing library for the `aind-ophys-movie-qc` Code Ocean capsule:
all the logic lives here with its own tests, and the capsule is a thin wrapper
that calls `aind_ophys_movie_qc_library.job.run`. Metadata reads, the
`Processing` builders and the QC document builders come from the shared
[`aind-pophys-metadata`](https://github.com/AllenNeuralDynamics/aind-pophys-metadata)
package, so they stay consistent across the pipeline's capsules.

## What it does

One task covers one plane. `run()`:

1. Reads the acquisition metadata through the shared `CoreMetadata` for the
   frame rate, the FOV scale factor and the logging identifiers.
2. Locates the registered movie (`*registered.h5`) under the input directory and
   derives the plane id from its filename.
3. Resolves the frame rate, preferring the rate motion correction recorded on
   the registered movie's `metadata` dataset, then the acquisition metadata,
   then `--frame_rate`. There is no default: a wrong frame rate silently rescales the
   epilepsy event widths and every per-second photon statistic.
4. Computes the statistics and renders the figures — intensity drift, epilepsy
   probability, SNR and d-prime, pixel-value distribution, ROI segmentation and
   photon flux, projection images.
5. Measures z-drift when `<plane>_z_stack_local.h5` is mounted, where `<plane>`
   is the plane id from step 2 (`VISp_0_registered.h5` gives `VISp_0`). Without
   it, z-drift is skipped with a warning.
6. Writes the raw metrics dump, a full `processing.json` and a full
   `quality_control.json`.

Movies shorter than `SHORT_MOVIE_MODE_THRESHOLD` frames (1001) enter
*short-movie mode*: the start/end stability windows do not exist, so intensity
drift, ROI segmentation, photon flux and d-prime are neither computed nor
reported. The metrics that would have carried them are omitted rather than
emitted with fabricated zeros.

## Modules

### `job.py`
Orchestration. `run()` is the entry point the capsule calls.

### `analysis.py`
The numeric routines and figure builders: `get_and_plot_epilepsy_probability`,
`get_spike_prominence_and_width`, `get_and_plot_basic_segmentation`,
`get_photon_gain_parameters`, `convert_intensity_into_photon_flux`,
`get_dprime_indicator`, `get_saturation_metrics`, `get_percentile_metrics`,
`get_simple_snr_metrics`, `get_percent_change_intensity`,
`subsample_and_crop_video`, the `plot_*` builders, `save_figure_to_storage` and
`make_output_directory`.

### `local_z_stack.py`
`LocalZStack`, which loop-averages the local z-stack and correlates the movie's
start and end mean images against every plane of it. `get_z_drift` returns the
drift in micrometres plus the images the figure needs.

### `qc.py`
Metric content for `quality_control.json`. Metric-producing functions
(`build_intensity_metric`, `build_epilepsy_metric`,
`build_event_detection_metric`, `build_pixel_distribution_metric`,
`build_photon_metric`, `build_zdrift_metric`) return
metrics; `collect_metrics` applies the run's gating and accumulates them, and
`build_quality_control` plus `write_quality_control_json` write one document per
plane. `intensity_status` and `saturation_status` hold the automatic thresholds.

### `settings.py`
`MovieQCSettings`, a `pydantic-settings` class parsed from the command line or
`MOVIE_QC_`-prefixed environment variables. The capsule's Code Ocean app panel
is generated from this file with `auto-app-panel`.

### `utils/metadata_utils.py`
Acquisition-metadata reads. `build_movie_qc_metadata` asks the shared
`CoreMetadata` for each field and returns the frozen `MovieQCMetadata` view.
`read_movie_frame_rate` reads the rate motion correction recorded on the
registered movie, and `resolve_frame_rate` closes the chain. Also `build_code`.

### `utils/image_utils.py`
PIL helpers — `combine_images_vertically`, `save_image` and
`build_combined_roi_photon_image` — that stack the segmentation and Poisson
figures into the composite the photon-detection metric references.

## Settings

| Field | Default | Purpose |
|---|---|---|
| `input_dir` | `../data/` | Motion-correction results and raw asset metadata, searched recursively. |
| `output_dir` | `../results/` | Results root. |
| `start_frame` | `1` | First frame of the movie block used for the main analysis. |
| `end_frame` | `10000` | End of that block (exclusive). |
| `crop` | `[30, 30]` | Pixels trimmed from each edge, as the JSON list `"[px_y, px_x]"`. |
| `max_pixel_range` | `18423` | Pixel value at or above which saturation is counted. |
| `min_pixel_range` | `0` | Pixel value at or below which under-saturation is counted. |
| `decay_time` | `0.2` | Reporter half decay time in seconds (GCaMP6f). |
| `dff_single_event_size` | `0.15` | Reporter single-event dF/F size (GCaMP6f). |
| `frame_rate` | from metadata | Override, used only when the resolution chain finds nothing. |

`crop` is declared `Optional[list]` and coerced by a validator: a JSON string is
the only form a Code Ocean app panel can emit for a pair.

## Outputs

Per plane, under `<output_dir>/<plane>/movie_qc/`, where `<base>` is the
registered movie's stem:

- `<base>_metrics.json` — the raw metrics dump, minus the bulky per-pixel
  arrays.
- `<base>_mean_projection_image.png`, `<base>_physio_intensity_plot.png`,
  `<base>_physio_intensity_hist.png`, `<base>_physio_poisson_plot.png` and
  `<base>_epilepsy_probability.png`.
- Long movies only: `<base>_start_projection_image.png`,
  `<base>_end_projection_image.png`, `<base>_basic_segmentation_image.png` and
  `<plane>_roi_and_photon_metrics.png`.
- With a local z-stack: `<base>_zdrift.png` and `<zstack>_reg.h5`, the
  loop-averaged registered z-stack.
- `processing.json` — a full `Processing` document. The
  `AssetPath ... does not exist` warning on it is expected and cosmetic:
  `output_path` is relative to the results root while the file sits two levels
  deeper.
- `quality_control.json` — a full `QualityControl` document. Up to six
  metrics, each name prefixed with the plane id and each tagged into its own
  `evaluation` group.

Nothing is written to the results root: `main.nf` publishes with `saveAs`
flattening to basename, so root-level files from concurrent per-plane tasks
would collide and abort the run.

## Level of Support
Please indicate a level of support:
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using [uv](https://docs.astral.sh/uv/), run
```bash
uv sync
```

Requires Python >=3.10,<3.13: `aind-pophys-metadata` declares that bound.

## Tests

Tests are `unittest`. The gate CI runs, replicated locally:

```bash
flake8 .
interrogate --verbose .
coverage run -m unittest discover
coverage report
```

`analysis.py`, `local_z_stack.py` and `job.py` are omitted from coverage — the
heavy movie compute, the z-stack registration, and orchestration only a real
Code Ocean data run exercises end to end. The remaining modules are held at 100%
rather than diluting the gate to an average.
