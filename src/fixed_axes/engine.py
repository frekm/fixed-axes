import matplotlib.pyplot as plt
import matplotlib.layout_engine as mlayout
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._fixed_axes_layout import do_fixed_axes_layout


class FixedAxesLayoutEngine(mlayout.LayoutEngine):
    _adjust_compatible = False
    _colorbar_gridspec = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._params = {}
        self._is_executing = False

    def execute(self, fig: Figure):
        if self._is_executing:
            return

        self._is_executing = True

        try:
            do_fixed_axes_layout(fig)

        finally:
            self._is_executing = False
