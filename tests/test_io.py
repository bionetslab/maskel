"""Tests for maskel._io."""

import csv

import numpy as np
import pytest
from skan import Skeleton, summarize

from maskel._io import save_analysis_outputs
from maskel.config import OutputConfig
from maskel.pipeline import AnalysisResult, ObjectResult


class TestSaveAnalysisOutputs:
    """Tests for save_analysis_outputs, the top-level writer."""

    @staticmethod
    def _result(
        *,
        summary: bool = True,
        radius: bool = False,
        branches: bool = False,
        nodes: bool = False,
    ) -> AnalysisResult:
        skel = np.eye(10, dtype=np.uint8)
        feat = (
            {"n_branches": 4.0, "n_junctions": 1.0, "object_id": 1} if summary else {}
        )
        rad = np.ones((10, 10), dtype=np.float64) if radius else None
        brecs = (
            [{"id": i, "len": float(i * 2), "object_id": 1} for i in range(2)]
            if branches
            else []
        )
        nrecs = (
            [{"id": i, "deg": i + 2, "object_id": 1} for i in range(2)] if nodes else []
        )
        return AnalysisResult(
            skeleton=skel,
            objects=[
                ObjectResult(
                    object_id=1,
                    offset=(0, 0),
                    summary_features=feat,
                    branch_records=brecs,
                    node_records=nrecs,
                )
            ],
            radius_matrix=rad,
        )

    # -- skeleton output ---------------------------------------------------

    def test_default_skeleton_npy_and_summary(self, tmp_path):
        save_analysis_outputs(tmp_path, "img", self._result(), OutputConfig())
        d = tmp_path / "img"
        assert d.is_dir()
        assert (d / "img_skeleton.npy").exists()
        assert (d / "img_summary.csv").exists()
        assert not (d / "img_skeleton.png").exists()

    def test_skeleton_png_only(self, tmp_path):
        cfg = OutputConfig(write_skeleton_npy=False, write_skeleton_png=True)
        save_analysis_outputs(tmp_path, "img", self._result(), cfg)
        d = tmp_path / "img"
        assert (d / "img_skeleton.png").exists()
        assert not (d / "img_skeleton.npy").exists()

    def test_skeleton_both_formats(self, tmp_path):
        cfg = OutputConfig(write_skeleton_npy=True, write_skeleton_png=True)
        save_analysis_outputs(tmp_path, "img", self._result(), cfg)
        d = tmp_path / "img"
        assert (d / "img_skeleton.npy").exists()
        assert (d / "img_skeleton.png").exists()

    def test_skeleton_neither_format(self, tmp_path):
        cfg = OutputConfig(write_skeleton_npy=False, write_skeleton_png=False)
        save_analysis_outputs(tmp_path, "img", self._result(), cfg)
        d = tmp_path / "img"
        assert not (d / "img_skeleton.npy").exists()
        assert not (d / "img_skeleton.png").exists()

    def test_3d_skeleton_with_png_raises(self, tmp_path):
        result = AnalysisResult(
            skeleton=np.ones((4, 4, 4), dtype=np.uint8),
            objects=[],
        )
        cfg = OutputConfig(write_skeleton_npy=False, write_skeleton_png=True)
        with pytest.raises(ValueError, match="PNG skeleton output"):
            save_analysis_outputs(tmp_path, "vol", result, cfg)

    # -- branch CSV --------------------------------------------------------

    def test_saves_branch_csv(self, tmp_path):
        cfg = OutputConfig(write_branch_csv=True)
        save_analysis_outputs(tmp_path, "img", self._result(branches=True), cfg)
        with open(tmp_path / "img" / "img_branches.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["id"] == "0"

    def test_skips_branch_csv_when_no_records(self, tmp_path):
        cfg = OutputConfig(write_branch_csv=True)
        save_analysis_outputs(tmp_path, "img", self._result(branches=False), cfg)
        assert not (tmp_path / "img" / "img_branches.csv").exists()

    def test_skips_branch_csv_when_disabled(self, tmp_path):
        cfg = OutputConfig(write_branch_csv=False)
        save_analysis_outputs(tmp_path, "img", self._result(branches=True), cfg)
        assert not (tmp_path / "img" / "img_branches.csv").exists()

    # -- node CSV ----------------------------------------------------------

    def test_saves_node_csv(self, tmp_path):
        cfg = OutputConfig(write_node_csv=True)
        save_analysis_outputs(tmp_path, "img", self._result(nodes=True), cfg)
        with open(tmp_path / "img" / "img_nodes.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["deg"] == "2"

    def test_skips_node_csv_when_no_records(self, tmp_path):
        cfg = OutputConfig(write_node_csv=True)
        save_analysis_outputs(tmp_path, "img", self._result(nodes=False), cfg)
        assert not (tmp_path / "img" / "img_nodes.csv").exists()

    def test_skips_node_csv_when_disabled(self, tmp_path):
        cfg = OutputConfig(write_node_csv=False)
        save_analysis_outputs(tmp_path, "img", self._result(nodes=True), cfg)
        assert not (tmp_path / "img" / "img_nodes.csv").exists()

    # -- radius ------------------------------------------------------------

    def test_saves_radius(self, tmp_path):
        cfg = OutputConfig(write_radius=True)
        save_analysis_outputs(tmp_path, "img", self._result(radius=True), cfg)
        path = tmp_path / "img" / "img_radius.npy"
        assert path.exists()
        assert np.load(path).dtype == np.float64

    def test_skips_radius_when_none(self, tmp_path):
        cfg = OutputConfig(write_radius=True)
        save_analysis_outputs(tmp_path, "img", self._result(radius=False), cfg)
        assert not (tmp_path / "img" / "img_radius.npy").exists()

    def test_skips_radius_when_disabled(self, tmp_path):
        cfg = OutputConfig(write_radius=False)
        save_analysis_outputs(tmp_path, "img", self._result(radius=True), cfg)
        assert not (tmp_path / "img" / "img_radius.npy").exists()

    # -- summary CSV -------------------------------------------------------

    def test_summary_csv_content(self, tmp_path):
        save_analysis_outputs(
            tmp_path, "img", self._result(summary=True), OutputConfig()
        )
        with open(tmp_path / "img" / "img_summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["image"] == "img"
        assert rows[0]["n_branches"] == "4.0"
        assert rows[0]["object_id"] == "1"

    def test_skips_summary_when_empty_features(self, tmp_path):
        save_analysis_outputs(
            tmp_path, "img", self._result(summary=False), OutputConfig()
        )
        assert not (tmp_path / "img" / "img_summary.csv").exists()

    def test_skips_summary_when_write_summary_false(self, tmp_path):
        save_analysis_outputs(
            tmp_path,
            "img",
            self._result(summary=True),
            OutputConfig(),
            write_summary=False,
        )
        assert not (tmp_path / "img" / "img_summary.csv").exists()

    def test_skips_summary_when_config_disabled(self, tmp_path):
        cfg = OutputConfig(write_summary_csv=False)
        save_analysis_outputs(tmp_path, "img", self._result(summary=True), cfg)
        assert not (tmp_path / "img" / "img_summary.csv").exists()

    # -- all outputs -------------------------------------------------------

    def test_all_outputs_enabled(self, tmp_path):
        result = self._result(summary=True, radius=True, branches=True, nodes=True)
        cfg = OutputConfig(
            write_skeleton_npy=True,
            write_skeleton_png=True,
            write_branch_csv=True,
            write_node_csv=True,
            write_radius=True,
            write_summary_csv=True,
        )
        save_analysis_outputs(tmp_path, "img", result, cfg)
        d = tmp_path / "img"
        assert (d / "img_skeleton.npy").exists()
        assert (d / "img_skeleton.png").exists()
        assert (d / "img_branches.csv").exists()
        assert (d / "img_nodes.csv").exists()
        assert (d / "img_radius.npy").exists()
        assert (d / "img_summary.csv").exists()

    def test_nothing_enabled_creates_empty_dir(self, tmp_path):
        cfg = OutputConfig(
            write_skeleton_npy=False,
            write_skeleton_png=False,
            write_branch_csv=False,
            write_node_csv=False,
            write_radius=False,
            write_summary_csv=False,
        )
        save_analysis_outputs(tmp_path, "img", self._result(summary=True), cfg)
        d = tmp_path / "img"
        assert d.is_dir()
        assert list(d.iterdir()) == []

    # -- edge cases --------------------------------------------------------

    def test_existing_dir_is_reused(self, tmp_path):
        d = tmp_path / "img"
        d.mkdir()
        (d / "stale.txt").touch()
        save_analysis_outputs(tmp_path, "img", self._result(), OutputConfig())
        assert (d / "stale.txt").exists()
        assert (d / "img_skeleton.npy").exists()

    def test_base_name_with_spaces(self, tmp_path):
        save_analysis_outputs(tmp_path, "my img", self._result(), OutputConfig())
        d = tmp_path / "my img"
        assert d.is_dir()
        assert (d / "my img_skeleton.npy").exists()

    # -- graphml -----------------------------------------------------------

    @pytest.fixture
    def graph_result(self, cross_skel) -> AnalysisResult:
        graph = Skeleton(cross_skel)
        branch_data = summarize(graph, separator="-")
        return AnalysisResult(
            skeleton=cross_skel,
            objects=[
                ObjectResult(
                    object_id=1,
                    offset=(0, 0),
                    summary_features={
                        "num_nodes": float(len(branch_data)),
                        "object_id": 1,
                    },
                    branch_records=[],
                    node_records=[],
                    graph=graph,
                    branch_data=branch_data,
                )
            ],
            radius_matrix=np.ones_like(cross_skel, dtype=np.float64) * 2.0,
        )

    def test_saves_graphml(self, tmp_path, graph_result):
        cfg = OutputConfig(write_graphml=True)
        save_analysis_outputs(tmp_path, "img", graph_result, cfg)
        assert (tmp_path / "img" / "img_1_graph.graphml").exists()

    def test_skips_graphml_when_disabled(self, tmp_path, graph_result):
        cfg = OutputConfig(write_graphml=False)
        save_analysis_outputs(tmp_path, "img", graph_result, cfg)
        assert not (tmp_path / "img" / "img_1_graph.graphml").exists()

    def test_skips_graphml_when_graph_missing(self, tmp_path):
        cfg = OutputConfig(write_graphml=True)
        save_analysis_outputs(tmp_path, "img", self._result(), cfg)
        assert not (tmp_path / "img" / "img_1_graph.graphml").exists()
