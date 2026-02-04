import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from typing import Literal, cast

MM_PER_INCH = 25.4


def set_axes_size(
    size: tuple[float, float],
    ax: Axes | None = None,
    unit: Literal["mm", "inch"] = "inch",
) -> None:
    valid_units = "mm", "inch"
    if unit not in valid_units:
        raise ValueError(f"{unit=}, but it must be in {valid_units}")

    ax = ax or plt.gca()
    fig = ax.get_figure(root=True)
    if fig is None:
        raise ValueError("ax must be part of a figure")

    if unit == "mm":
        size = size[0] / MM_PER_INCH, size[1] / MM_PER_INCH

    fw, fh = fig.get_size_inches()

    new_width, new_height = fw * size[0], fh * size[1]

    anchor = ax.get_anchor()
    if isinstance(anchor, str) and anchor in mtransforms.Bbox.coefs:
        anchor = mtransforms.Bbox.coefs[anchor]
    anchor = cast(tuple[float, float], anchor)

    ax.stale = True
