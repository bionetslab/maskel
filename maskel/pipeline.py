"""Shared analysis pipeline used by the CLI (and by napari-maskel, which wraps it)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi
from skan import summarize

from maskel.config import PipelineConfig
from maskel.features import (
    build_vessel_graph,
    compute_radii,
    compute_tortuosity,
    extract_node_features,
    extract_vessel_features,
    per_segment_radii,
)
from maskel.junction_cleanup import collapse_triangle_junctions
from maskel.spur_pruning import prune_short_spurs
from maskel.thin import lee94_thin

if TYPE_CHECKING:
    from pandas import DataFrame
    from skan import Skeleton


@dataclass
class ObjectResult:
    """Analysis outputs for one object, tagged and positioned in the full image.

    ``summary_features``/``branch_records``/``node_records`` are already
    tagged with ``object_id`` (node coordinates already offset into the full
    image). ``graph``/``branch_data``/``radius_matrix`` remain in this
    object's own local (crop) coordinates - kept separate per object, rather
    than merged into one graph, since objects are skeletonized independently
    and touching-but-distinct objects must stay topologically separate.
    *offset* is the crop's origin in the full image, for anything derived
    from ``graph``/``branch_data`` that needs global coordinates (e.g.
    ``graph.path_coordinates(i) + offset``).
    """

    object_id: int
    offset: tuple[int, ...]
    summary_features: dict[str, float]
    branch_records: list[dict[str, object]]
    node_records: list[dict[str, object]]
    graph: Skeleton | None = None
    branch_data: DataFrame | None = None
    radius_matrix: np.ndarray | None = None


@dataclass
class AnalysisResult:
    """Container for single-image analysis outputs.

    Deliberately napari-free: napari-maskel builds its own LayerDataTuples
    from these fields rather than this package returning them directly.

    An input with more than one distinct nonzero value is treated as an
    instance segmentation map (each value its own object); a plain binary
    mask is treated as a single implicit object with id 1. ``objects`` has
    one entry per object; ``skeleton``/``radius_matrix``/
    ``preprocessed_binary`` are stitched back to the full input shape, since
    (unlike per-object lists) that stitching is real work only worth doing
    once, centrally, regardless of how many consumers want a full-image view.
    """

    skeleton: np.ndarray
    objects: list[ObjectResult]
    radius_matrix: np.ndarray | None = None
    preprocessed_binary: np.ndarray | None = None

    @property
    def summary_features(self) -> list[dict[str, float]]:
        """One dict per object that has any (empty when summary is disabled)."""
        return [obj.summary_features for obj in self.objects if obj.summary_features]

    @property
    def branch_records(self) -> list[dict[str, object]]:
        """All objects' branch records, flattened."""
        return [record for obj in self.objects for record in obj.branch_records]

    @property
    def node_records(self) -> list[dict[str, object]]:
        """All objects' node records, flattened."""
        return [record for obj in self.objects for record in obj.node_records]


def preprocess_binary(
    binary: np.ndarray,
    closing_iterations: int = 0,
    fill_holes: bool = False,
    max_hole_size: int = 0,
) -> np.ndarray:
    """Apply morphological preprocessing to a binary mask.

    Parameters
    ----------
    binary : ndarray
        Binary uint8 array (0/1).
    closing_iterations : int
        Number of binary closing iterations (0 = skip).
    fill_holes : bool
        Whether to fill enclosed background regions.
    max_hole_size : int
        Max hole size in voxels to fill (0 = unlimited).

    Returns
    -------
    ndarray
        Preprocessed binary array.
    """
    if closing_iterations > 0:
        structure = ndi.generate_binary_structure(binary.ndim, 1)
        binary = ndi.binary_closing(
            binary, structure=structure, iterations=closing_iterations
        ).astype(binary.dtype)

    if fill_holes:
        before_fill = binary.copy() if max_hole_size > 0 else None
        binary = ndi.binary_fill_holes(binary).astype(binary.dtype)

        if max_hole_size > 0 and before_fill is not None:
            diff = binary.astype(np.int8) - before_fill.astype(np.int8)
            filled = diff > 0
            if filled.any():
                labels, _ = ndi.label(filled)
                sizes = np.bincount(labels.ravel())
                big = sizes > max_hole_size
                big[0] = False
                revert = big[labels]
                binary[revert] = 0

    return binary


def _iter_object_crops(
    mask: np.ndarray,
) -> list[tuple[int, tuple[slice, ...], np.ndarray]]:
    """Split a segmentation mask into per-object, bounding-box-cropped binary arrays.

    A plain binary mask (at most one distinct nonzero value) is treated as a
    single implicit object with id 1. A mask with more than one distinct
    nonzero value is treated as an instance segmentation map, where each
    distinct value is its own object - this is how touching-but-distinct
    objects end up correctly separated rather than merged into one skeleton.

    Each crop is the object's bounding box padded by one voxel of background
    (clamped to the array bounds), so local operations (thinning, closing,
    EDT) aren't affected by the crop boundary coinciding with the object's
    own boundary.

    Returns
    -------
    list of (object_id, bbox, binary_crop)
        *bbox* is a tuple of slices into *mask* (the padded bounding box).
        *binary_crop* is the object's mask within *bbox*, as a uint8 array.

    Notes
    -----
    Costs that scale with the *whole* mask rather than each object's own
    crop are kept to the minimum needed to find object bounding boxes: one
    ``ndi.find_objects`` pass, and a dtype cast only when *mask* isn't
    already an integer array. In particular this avoids ``np.unique``
    (which would sort every foreground pixel's value) purely to tell a
    plain binary mask apart from a labeled one - a mask with exactly one
    surviving object *is* that binary-mask case, whatever its raw
    foreground value, so the same ``find_objects`` pass answers both
    questions at once.
    """
    if not mask.any():
        return []

    labels = mask if np.issubdtype(mask.dtype, np.integer) else mask.astype(np.int64)
    bboxes = ndi.find_objects(labels)
    raw_ids = [i + 1 for i, bbox in enumerate(bboxes) if bbox is not None]

    # A mask with only one object - including a plain binary mask, whatever
    # its raw foreground value - is reported as object id 1; a genuine
    # multi-object instance segmentation keeps each object's own label value.
    report_ids = [1] if len(raw_ids) == 1 else raw_ids

    crops = []
    for raw_id, object_id in zip(raw_ids, report_ids, strict=True):
        bbox = bboxes[raw_id - 1]
        padded = tuple(
            slice(max(s.start - 1, 0), min(s.stop + 1, dim))
            for s, dim in zip(bbox, mask.shape, strict=True)
        )
        binary_crop = (labels[padded] == raw_id).astype(np.uint8)
        crops.append((object_id, padded, binary_crop))

    return crops


@dataclass
class _ObjectResult:
    """Analysis outputs for a single already-cropped, already-binary object,
    in that crop's own local coordinates and not yet tagged with an object id
    - see ``ObjectResult`` for the packaged, positioned, public equivalent.
    """

    skeleton: np.ndarray
    summary_features: dict[str, float]
    branch_records: list[dict[str, object]]
    node_records: list[dict[str, object]]
    graph: Skeleton | None = None
    branch_data: DataFrame | None = None
    radius_matrix: np.ndarray | None = None
    preprocessed_binary: np.ndarray | None = None


def _analyze_single_object(binary: np.ndarray, config: PipelineConfig) -> _ObjectResult:
    """Run the skeletonization + extraction pipeline for one binary object.

    *binary* is expected to already be a single object's cropped binary mask
    (see ``_iter_object_crops``); this is the same pipeline the whole image
    used to go through as a single unit before per-object splitting.
    """
    # -- optional: morphological preprocessing -------------------------
    preprocessed_binary: np.ndarray | None = None
    if config.extraction.closing_iterations > 0 or config.extraction.fill_holes:
        binary = preprocess_binary(
            binary,
            closing_iterations=config.extraction.closing_iterations,
            fill_holes=config.extraction.fill_holes,
            max_hole_size=config.extraction.max_hole_size,
        )
        preprocessed_binary = binary

    skeleton = lee94_thin(binary)

    if not skeleton.any():
        return _ObjectResult(
            skeleton=skeleton,
            summary_features={},
            branch_records=[],
            node_records=[],
            preprocessed_binary=preprocessed_binary,
        )

    # -- optional: collapse triangle junction artifacts -----------------
    # (requires EDT; done on the original skeleton before graph building)
    if config.extraction.junction_cleanup:
        rm_temp, _ = compute_radii(binary, skeleton)
        skeleton = collapse_triangle_junctions(
            skeleton,
            radius_matrix=rm_temp,
            threshold_factor=config.extraction.cleanup_threshold_factor,
        )

    # -- build graph & branch data on the (potentially cleaned) skeleton -
    graph = build_vessel_graph(skeleton)

    # -- optional: prune short spur branches -----------------------------
    # (a spur: one node is an endpoint (degree 1), the other a junction
    # (degree > 1), and the branch is shorter than the configured
    # threshold - usually a thinning artifact rather than a real vessel
    # tip. Pruning changes the skeleton's pixels, so the graph is rebuilt
    # from scratch afterwards.)
    if config.extraction.prune_spurs:
        # Pruning may expose new short spurs (an endpoint's parent junction
        # can become an endpoint once its own spur is removed), so the
        # operation is repeated on its own output up to spur_iterations times.
        for _ in range(max(1, config.extraction.spur_iterations)):
            pre_prune_branch_data = summarize(graph, separator="-")
            skeleton = prune_short_spurs(
                skeleton,
                graph,
                pre_prune_branch_data,
                min_length=config.extraction.min_spur_length,
            )
            if not skeleton.any():
                return _ObjectResult(
                    skeleton=skeleton,
                    summary_features={},
                    branch_records=[],
                    node_records=[],
                    preprocessed_binary=preprocessed_binary,
                )
            graph = build_vessel_graph(skeleton)

    # -- optional: vessel radius (EDT on the final skeleton) ------------
    radius_matrix = None
    radius_stats = None
    if config.extraction.vessel_radius:
        radius_matrix, radius_stats = compute_radii(binary, skeleton)

    branch_data = summarize(graph, separator="-")

    if radius_matrix is not None and not branch_data.empty:
        per_seg = per_segment_radii(radius_matrix, graph, len(branch_data))
        for key, arr in per_seg.items():
            branch_data[key] = arr

        if radius_stats is not None:
            radius_stats["mean_segment_volume"] = float(
                np.nanmean(branch_data["volume"])
            )
            radius_stats["mean_surface_area"] = float(
                np.nanmean(branch_data["surface_area"])
            )

    if not branch_data.empty:
        euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)
        branch_dist = branch_data["branch-distance"].to_numpy(dtype=float)
        tortuosity = compute_tortuosity(branch_dist, euclidean)
        branch_data["tortuosity"] = tortuosity
        straightness = np.full_like(branch_dist, np.nan, dtype=float)
        valid = branch_dist > 0
        straightness[valid] = euclidean[valid] / branch_dist[valid]
        branch_data["straightness"] = straightness

    summary_features: dict[str, float] = {}
    if config.extraction.summary:
        summary_features = extract_vessel_features(
            skeleton,
            graph,
            branch_data,
            binary=binary,
            include_fractal=config.extraction.fractal_dimension,
            radius_stats=radius_stats,
        )

    branch_records: list[dict[str, object]] = []
    if config.extraction.branches and not branch_data.empty:
        branch_records = branch_data.to_dict(orient="records")

    node_records: list[dict[str, object]] = []
    if config.extraction.nodes:
        node_records = extract_node_features(
            graph,
            branch_data,
            radius_matrix=radius_matrix,
        )

    return _ObjectResult(
        skeleton=skeleton,
        summary_features=summary_features,
        branch_records=branch_records,
        node_records=node_records,
        graph=graph,
        branch_data=branch_data,
        radius_matrix=radius_matrix,
        preprocessed_binary=preprocessed_binary,
    )


def analyze_segmentation_mask(
    mask: np.ndarray, config: PipelineConfig
) -> AnalysisResult:
    """Run the full skeletonization + extraction pipeline for one segmentation mask.

    Accepts either a plain binary segmentation mask or a multi-object
    instance segmentation map (more than one distinct nonzero value). Each
    object is cropped to its own (padded) bounding box and processed fully
    independently - this is a performance optimization for binary input, and
    is what correctly separates touching-but-distinct objects for labeled
    input, rather than merging them into one skeleton. Every object's
    records are tagged with the object id it came from (``1`` for a plain
    binary mask; see ``AnalysisResult``/``ObjectResult``).

    Parameters
    ----------
    mask : ndarray
        Input segmentation mask. Non-zero values are treated as foreground;
        see above for how multiple distinct nonzero values are interpreted.
    config : PipelineConfig
        Full pipeline configuration.
    """
    object_crops = _iter_object_crops(mask)

    if not object_crops:
        return AnalysisResult(
            skeleton=np.zeros(mask.shape, dtype=np.uint8),
            objects=[],
        )

    want_radius = config.extraction.vessel_radius
    want_preprocessed = (
        config.extraction.closing_iterations > 0 or config.extraction.fill_holes
    )

    full_skeleton = np.zeros(mask.shape, dtype=np.uint8)
    full_radius = np.zeros(mask.shape, dtype=np.float64) if want_radius else None
    full_preprocessed = (
        np.zeros(mask.shape, dtype=np.uint8) if want_preprocessed else None
    )

    ndim = mask.ndim
    objects: list[ObjectResult] = []

    for object_id, bbox, crop in object_crops:
        obj = _analyze_single_object(crop, config)
        offset = tuple(s.start for s in bbox)

        full_skeleton[bbox] = np.maximum(full_skeleton[bbox], obj.skeleton)
        if full_radius is not None and obj.radius_matrix is not None:
            full_radius[bbox] = np.maximum(full_radius[bbox], obj.radius_matrix)
        if full_preprocessed is not None and obj.preprocessed_binary is not None:
            full_preprocessed[bbox] = np.maximum(
                full_preprocessed[bbox], obj.preprocessed_binary
            )

        summary_features = (
            {**obj.summary_features, "object_id": object_id}
            if config.extraction.summary
            else {}
        )

        branch_records = [
            {**record, "object_id": object_id} for record in obj.branch_records
        ]

        node_records = []
        for record in obj.node_records:
            record = dict(record)
            for d in range(ndim):
                record[f"coord_{d}"] = record[f"coord_{d}"] + offset[d]
            record["object_id"] = object_id
            node_records.append(record)

        objects.append(
            ObjectResult(
                object_id=object_id,
                offset=offset,
                summary_features=summary_features,
                branch_records=branch_records,
                node_records=node_records,
                graph=obj.graph,
                branch_data=obj.branch_data,
                radius_matrix=obj.radius_matrix,
            )
        )

    return AnalysisResult(
        skeleton=full_skeleton,
        objects=objects,
        radius_matrix=full_radius,
        preprocessed_binary=full_preprocessed,
    )
