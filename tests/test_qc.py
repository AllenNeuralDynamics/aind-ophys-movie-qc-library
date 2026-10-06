"""Tests for quality_control.json generation."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
from aind_data_schema.core.quality_control import QCMetric, QualityControl, Stage, Status
from aind_data_schema_models.modalities import Modality

from aind_ophys_movie_qc_library import qc
from tests import fixtures

UNIQUE_ID = fixtures.UNIQUE_ID

REFERENCES = {
    qc.REF_EPILEPSY: f"{UNIQUE_ID}/movie_qc/{UNIQUE_ID}_registered_epilepsy_probability.png",
    qc.REF_INTENSITY_PLOT: f"{UNIQUE_ID}/movie_qc/{UNIQUE_ID}_registered_physio_intensity_plot.png",
    qc.REF_INTENSITY_HIST: f"{UNIQUE_ID}/movie_qc/{UNIQUE_ID}_registered_physio_intensity_hist.png",
    qc.REF_COMBINED: f"{UNIQUE_ID}/movie_qc/{UNIQUE_ID}_roi_and_photon_metrics.png",
    qc.REF_ZDRIFT: f"{UNIQUE_ID}/movie_qc/{UNIQUE_ID}_registered_zdrift.png",
    qc.REF_PROJECTIONS: f"{UNIQUE_ID}/movie_qc/{UNIQUE_ID}_registered_mean_projection_image.png",
}
PROJECTIONS = [REFERENCES[qc.REF_PROJECTIONS]]


class TestAutomaticStatuses(unittest.TestCase):
    """Each automatic status flips exactly at its named threshold."""

    def test_intensity_thresholds(self) -> None:
        """PASS below 10%, PENDING to 20%, FAIL at or above 20%."""
        self.assertEqual(qc.intensity_status(-9.9), Status.PASS)
        self.assertEqual(qc.intensity_status(10.0), Status.PENDING)
        self.assertEqual(qc.intensity_status(-19.9), Status.PENDING)
        self.assertEqual(qc.intensity_status(-20.0), Status.FAIL)

    def test_saturation_thresholds(self) -> None:
        """PASS below 800 pixels, PENDING to 1000, FAIL at or above 1000."""
        self.assertEqual(qc.saturation_status(799), Status.PASS)
        self.assertEqual(qc.saturation_status(800), Status.PENDING)
        self.assertEqual(qc.saturation_status(1000), Status.FAIL)


class TestMetricSharedContract(unittest.TestCase):
    """Every metric must satisfy the project-wide QC contract."""

    def setUp(self) -> None:
        """Collect the full metric set for a long movie with a z-stack."""
        self.metrics = qc.collect_metrics(
            unique_id=UNIQUE_ID,
            metrics=fixtures.raw_metrics(),
            references=REFERENCES,
            projections=PROJECTIONS,
            short_movie_mode=False,
            zdrift=fixtures.zdrift_result(),
        )

    def test_every_metric_carries_evaluation_and_type_tags(self) -> None:
        """default_grouping is ["evaluation"], so the tag must be present."""
        for metric in self.metrics:
            with self.subTest(metric=metric.name):
                self.assertIn("evaluation", metric.tags)
                self.assertEqual(metric.tags["type"], qc.METRIC_TYPE)

    def test_every_metric_seeds_a_status(self) -> None:
        """An empty status_history is rejected at construction."""
        for metric in self.metrics:
            with self.subTest(metric=metric.name):
                self.assertTrue(metric.status_history)
                self.assertEqual(metric.status_history[0].evaluator, qc.EVALUATOR)

    def test_every_metric_names_its_plane(self) -> None:
        """Every name carries the plane id, so multiplane metrics stay distinct."""
        for metric in self.metrics:
            with self.subTest(metric=metric.name):
                self.assertTrue(metric.name.startswith(UNIQUE_ID))

    def test_every_metric_has_a_description(self) -> None:
        """The reviewer prose is the point of these metrics."""
        for metric in self.metrics:
            with self.subTest(metric=metric.name):
                self.assertTrue(metric.description)

    def test_modality_and_stage(self) -> None:
        """Every metric is pophys processing-stage QC."""
        for metric in self.metrics:
            with self.subTest(metric=metric.name):
                self.assertEqual(metric.modality, Modality.POPHYS)
                self.assertEqual(metric.stage, Stage.PROCESSING)

    def test_every_supplied_reference_reaches_its_metric(self) -> None:
        """A builder that forgets to forward ``reference`` silently drops the
        figure link, which is invisible until a reviewer opens the portal."""
        expected = {
            f"{UNIQUE_ID} Epilepsy probability": REFERENCES[qc.REF_EPILEPSY],
            f"{UNIQUE_ID} Pixel value distribution": REFERENCES[qc.REF_INTENSITY_HIST],
            f"{UNIQUE_ID} Projection images": REFERENCES[qc.REF_PROJECTIONS],
            f"{UNIQUE_ID} Intensity stability": REFERENCES[qc.REF_INTENSITY_PLOT],
            f"{UNIQUE_ID} Photon detection statistics": REFERENCES[qc.REF_COMBINED],
            f"{UNIQUE_ID} Z-drift Analysis": REFERENCES[qc.REF_ZDRIFT],
        }
        actual = {metric.name: metric.reference for metric in self.metrics}
        for name, reference in expected.items():
            with self.subTest(metric=name):
                self.assertEqual(str(actual[name]), reference)
        self.assertIsNone(actual[f"{UNIQUE_ID} Event detection statistics"])

    def test_metric_names_are_unique(self) -> None:
        """Duplicate names in one document are unreadable in the portal."""
        names = [metric.name for metric in self.metrics]
        self.assertEqual(len(names), len(set(names)))


class TestIndividualMetrics(unittest.TestCase):
    """Values and statuses of each builder."""

    def test_intensity_metric_value_is_a_percent_string(self) -> None:
        """The value is a two-decimal percentage string."""
        metric = qc.build_intensity_metric(
            UNIQUE_ID, {"percent_change_intensity": np.float64(-3.256)}
        )
        self.assertEqual(metric.value, "-3.26%")
        self.assertEqual(metric.status_history[0].status, Status.PASS)

    def test_numpy_statistics_are_coerced_to_python_scalars(self) -> None:
        """Every statistic reaches the builders as a numpy scalar.

        ``QCMetric.value`` is typed ``Any``, so any numpy scalar validates;
        ``np.float64`` happens to subclass ``float``, but other numpy scalar
        types fail only at serialization. The builders' ``float()``/``int()``
        coercions keep every value a Python scalar, and only a numpy-typed
        metrics dict exercises them.
        """
        metrics = fixtures.raw_metrics()
        built = [
            qc.build_event_detection_metric(UNIQUE_ID, metrics),
            qc.build_pixel_distribution_metric(UNIQUE_ID, metrics),
            qc.build_photon_metric(UNIQUE_ID, metrics),
            qc.build_zdrift_metric(UNIQUE_ID, fixtures.zdrift_result()),
        ]
        for metric in built:
            with self.subTest(metric=metric.name):
                for name, value in metric.value.items():
                    self.assertNotIsInstance(value, np.generic, name)

    def test_epilepsy_metric_fails_on_any_probability(self) -> None:
        """Any non-zero probability is a FAIL."""
        passing = qc.build_epilepsy_metric(UNIQUE_ID, {"epilepsy_probability": 0.0})
        failing = qc.build_epilepsy_metric(UNIQUE_ID, {"epilepsy_probability": 0.01})
        self.assertEqual(passing.status_history[0].status, Status.PASS)
        self.assertEqual(failing.status_history[0].status, Status.FAIL)
        self.assertIsInstance(failing.value, float)

    def test_event_detection_metric_omits_dprime_on_short_movies(self) -> None:
        """A fabricated 0.0 would read as a real measurement."""
        full = qc.build_event_detection_metric(UNIQUE_ID, fixtures.raw_metrics())
        short = qc.build_event_detection_metric(
            UNIQUE_ID, fixtures.raw_metrics(), short_movie_mode=True
        )
        self.assertIn("Median ROI D-prime", full.value)
        self.assertNotIn("Median ROI D-prime", short.value)
        self.assertIn("SNR Mean", short.value)

    def test_pixel_distribution_metric_reports_five_values(self) -> None:
        """All five pixel statistics are present."""
        metric = qc.build_pixel_distribution_metric(UNIQUE_ID, fixtures.raw_metrics())
        self.assertEqual(len(metric.value), 5)
        self.assertEqual(metric.value["Saturated Pixels"], 12)

    def test_photon_metric_reports_ten_values(self) -> None:
        """All ten photon statistics are present."""
        metric = qc.build_photon_metric(UNIQUE_ID, fixtures.raw_metrics())
        self.assertEqual(len(metric.value), 10)
        self.assertEqual(metric.value["Photon Gain"], 2.5)

    def test_photon_metric_tolerates_missing_inputs(self) -> None:
        """An absent input reads as 0.0 rather than raising."""
        metric = qc.build_photon_metric(UNIQUE_ID, {})
        self.assertEqual(metric.value["Photon Gain"], 0.0)

    def test_zdrift_metric_passes_within_the_threshold(self) -> None:
        """Drift within the threshold is a PASS."""
        metric = qc.build_zdrift_metric(UNIQUE_ID, fixtures.zdrift_result(z_drift_um=4.0))
        self.assertEqual(metric.status_history[0].status, Status.PASS)
        self.assertEqual(len(metric.value), 10)
        self.assertEqual(metric.value["start_frame_corr"], 0.912)

    def test_zdrift_metric_fails_beyond_the_threshold(self) -> None:
        """Drift beyond the threshold is a FAIL."""
        metric = qc.build_zdrift_metric(UNIQUE_ID, fixtures.zdrift_result(z_drift_um=-25.0))
        self.assertEqual(metric.status_history[0].status, Status.FAIL)

    def test_zdrift_values_exclude_the_per_plane_arrays(self) -> None:
        """start_corr/end_corr are long and live in the raw metrics dump."""
        values = qc.zdrift_values(fixtures.zdrift_result())
        self.assertNotIn("start_corr", values)
        self.assertNotIn("end_corr", values)

    def test_projection_metric_lists_the_figures_it_adopts(self) -> None:
        """The metric references every projection figure that was written."""
        metric = qc.build_projection_metric(UNIQUE_ID, PROJECTIONS, PROJECTIONS[0])
        self.assertEqual(metric.value["projections"], PROJECTIONS)
        self.assertEqual(metric.status_history[0].status, Status.PASS)


class TestCollectMetrics(unittest.TestCase):
    """The collector owns all the gating so job.py stays thin."""

    def _collect(self, **kwargs: object) -> List[QCMetric]:
        """Collect with the long-movie, z-stack-present defaults."""
        options = {
            "short_movie_mode": False,
            "zdrift": fixtures.zdrift_result(),
        }
        options.update(kwargs)
        return qc.collect_metrics(
            unique_id=UNIQUE_ID,
            metrics=fixtures.raw_metrics(),
            references=REFERENCES,
            projections=PROJECTIONS,
            **options,
        )

    def test_full_run_emits_seven_metrics(self) -> None:
        """A full-length run emits all seven metrics."""
        self.assertEqual(len(self._collect()), 7)

    def test_short_movie_drops_intensity_and_photon_metrics(self) -> None:
        """Their inputs were never computed, so they must not be emitted."""
        names = [m.name for m in self._collect(short_movie_mode=True)]
        self.assertNotIn(f"{UNIQUE_ID} Intensity stability", names)
        self.assertNotIn(f"{UNIQUE_ID} Photon detection statistics", names)
        self.assertIn(f"{UNIQUE_ID} Epilepsy probability", names)
        self.assertIn(f"{UNIQUE_ID} Pixel value distribution", names)

    def test_no_zstack_drops_the_zdrift_metric(self) -> None:
        """Without a z-stack there is nothing for a reviewer to assess."""
        names = [m.name for m in self._collect(zdrift=None)]
        self.assertNotIn(f"{UNIQUE_ID} Z-drift Analysis", names)

    def test_missing_reference_yields_no_reference(self) -> None:
        """A figure that was not written must not fabricate a broken link."""
        metrics = qc.collect_metrics(
            unique_id=UNIQUE_ID,
            metrics=fixtures.raw_metrics(),
            references={},
            projections=[],
            short_movie_mode=True,
        )
        self.assertTrue(all(metric.reference is None for metric in metrics))

    def test_every_collected_item_is_a_qc_metric(self) -> None:
        """No CurationMetric or qcportal value type belongs here."""
        for metric in self._collect():
            self.assertIs(type(metric), QCMetric)


class TestQualityControlDocument(unittest.TestCase):
    """The full QualityControl document and its writer."""

    def _document(self) -> QualityControl:
        """Build a QualityControl over the full metric set."""
        metrics = qc.collect_metrics(
            unique_id=UNIQUE_ID,
            metrics=fixtures.raw_metrics(),
            references=REFERENCES,
            projections=PROJECTIONS,
            zdrift=fixtures.zdrift_result(),
        )
        return qc.build_quality_control(metrics)

    def test_build_uses_project_default_grouping(self) -> None:
        """The shared standard grouping is applied."""
        document = self._document()
        self.assertIsInstance(document, QualityControl)
        self.assertEqual(document.default_grouping, ["evaluation"])

    def test_status_derives_without_error(self) -> None:
        """An empty status_history anywhere would break this derivation."""
        self.assertTrue(self._document().status)

    def test_write_and_round_trip(self) -> None:
        """The written file validates back into a core v2 QualityControl."""
        document = self._document()
        with tempfile.TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(document, Path(tmp))
            self.assertTrue(written.exists())
            payload = json.loads(written.read_text())
            self.assertIn("describedBy", payload)
            reloaded = QualityControl.model_validate_json(written.read_text())
        self.assertEqual(len(reloaded.metrics), 7)


if __name__ == "__main__":
    unittest.main()
