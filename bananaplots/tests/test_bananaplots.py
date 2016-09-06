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
from numpy.random import normal

def test_bananas():
    numpy.random.seed(1234)
    s1 = bananaplots.GMMSurface(
            X=numpy.concatenate([normal(size=10000), normal(size=10000)]),
            Y=numpy.concatenate([normal(size=10000), normal(size=10000)])
        )
    s2 = bananaplots.GMMSurface(X=2 + normal(size=10000), Y=normal(size=10000))

    banana = bananaplots.Bananas()

    banana.add_surface(s1, label="S1", cmap=cm.Reds_r)
    banana.add_surface(s2, label="S2", cmap=cm.Blues_r)
    banana.add_surface(s1 + s2, label="S1 + S2", cmap=cm.Greens_r, compiler_options=dict(nc=2))

    banana.set_feature("X", range=(-6, 6))
    banana.set_feature("Y", range=(-6, 6))

    fig = Figure()
    ax = fig.add_subplot(223)
    banana.render(ax, "X", "Y")

    ax = fig.add_subplot(221)
    banana.render1d(ax, "X")

    ax = fig.add_subplot(224)
    banana.render1d(ax, "Y")

    handlers, labels = banana.get_legend_handlers_labels()
    fig.legend(handlers, labels, loc='center', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))

    canvas = FigureCanvasAgg(fig)

    fig.savefig("bananas.pdf")

if __name__ == '__main__':
    unittest.main()
