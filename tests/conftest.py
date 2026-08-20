import numpy as np
import pytest
from skan import Skeleton, summarize

from ._helpers import cross_image, cross_volume, loop_image


@pytest.fixture
def cross_skel() -> np.ndarray:
    return cross_image()


@pytest.fixture
def cross_graph(cross_skel):
    graph = Skeleton(cross_skel)
    return graph, summarize(graph, separator="-")


@pytest.fixture
def loop_graph():
    graph = Skeleton(loop_image())
    return graph, summarize(graph, separator="-")


@pytest.fixture
def cross_volume_graph():
    graph = Skeleton(cross_volume())
    return graph, summarize(graph, separator="-")
