from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from . import _utils as utils


class InvalidFigureError(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


def validate_figure(fig):
    if isinstance(fig, SubFigure):
        raise InvalidFigureError("FixedLayoutEngine cannot handle nested figures")
    gs = None
    for ax in fig._localaxes:
        if not ax.get_subplotspec() or not ax.get_in_layout():
            continue

        ss = ax.get_subplotspec()
        gs_ = ss.get_gridspec()
        if gs is None:
            gs = gs_
        elif gs_ != gs:
            msg = "FixedLayoutEngine cannot handle multiple gridspecs in figure"
            raise InvalidFigureError(msg)
    if gs is None:
        raise InvalidFigureError("Axes in figure need to be part of a gridspec")


def get_axes_grid(axes: list[Axes]) -> utils.Array[Axes]:
    for ax in axes:
        ss = ax.get_subplotspec()
        if ss is None or not ax.get_in_layout():
            continue
        gs = ss.get_gridspec()
        nrows, ncols = gs.get_geometry()
    rtn = np.empty((nrows, ncols), dtype=Axes)
    return rtn


def do_fixed_layout(fig):
    axes_grid = get_axes_grid(fig._localaxes)
