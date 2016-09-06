"""
Unit tests for Greeter class
"""
import unittest
import os
import bananaplots
import numpy
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter

from numpy.random import normal

def test_bananas():
    numpy.random.seed(1234)
    s1 = bananaplots.GMMSurface(
            X=numpy.concatenate([normal(size=10000), normal(size=10000)]),
            Y=numpy.concatenate([normal(size=10000), normal(size=10000)]),
            Z=numpy.concatenate([normal(size=10000), normal(size=10000)]),
        )
    s2 = bananaplots.GMMSurface(
            X=2 + normal(size=10000),
            Y=normal(size=10000), 
            Z=1 + normal(size=10000), 
            )

    banana = bananaplots.Bananas()

    banana.add_surface(s1, label="S1", cmap=cm.Reds_r)
    banana.add_surface(s2, label="S2", cmap=cm.Blues_r)
    banana.add_surface(s1 + s2, label="S1 + S2", cmap=cm.Greens_r, compiler_options=dict(nc=2))

    banana.set_feature("X", range=(-6, 6))
    banana.set_feature("Y", range=(-6, 6))
    banana.set_feature("Z", range=(-6, 6))

    fig = Figure()
    gs = GridSpec(3, 3, hspace=0, wspace=0)

    axes = {}
    for i, f in zip(range(3), "XYZ"):
        ax = fig.add_subplot(gs[i, i])
        banana.render1d(ax, f)
        ax.locator_params(axis='y', nbins=5)
        ax.yaxis.set_major_formatter(NullFormatter())
        axes[i, i] = ax

    for i, j in [(1, 0), (2, 0), (2, 1)]:
        ax = fig.add_subplot(gs[i, j])
        banana.render(ax, "XYZ"[j], "XYZ"[i])
        axes[i, j] = ax

    for (i, j), ax in axes.items():
        if i != 2:
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.get_label().set_visible(False)
        if j != 0:
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.get_label().set_visible(False)
        ax.locator_params(axis='y', prune='both')
        ax.locator_params(axis='x', prune='both')

    handlers, labels = banana.get_legend_handlers_labels()
    fig.legend(handlers, labels, loc='center', bbox_to_anchor=gs[0, -1].get_position(fig))

    canvas = FigureCanvasAgg(fig)
#    fig.tight_layout()
    fig.savefig("bananas.pdf")

if __name__ == '__main__':
    unittest.main()
