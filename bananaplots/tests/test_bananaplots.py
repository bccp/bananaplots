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
    s3 = s1 + s2
    f1 = s1.compile(['X', 'Y'])

    banana = bananaplots.Bananas()

    banana.add_surface(s1, label="S1", cmap=cm.Reds_r)
    banana.add_surface(s2, label="S2", cmap=cm.Blues_r)
    banana.add_surface(s3, label="S1 + S2", cmap=cm.Greens_r, compiler_options=dict(nc=2))

    banana.set_feature("X", range=(-6, 6))
    banana.set_feature("Y", range=(-6, 6))

    fig = Figure()
    ax = fig.add_subplot(111)
    banana.render(ax, "X", "Y")

    handlers, labels = banana.get_legend_handlers_labels()
    ax.legend(handlers, labels)
    canvas = FigureCanvasAgg(fig)
    fig.savefig("bananas.pdf")

if __name__ == '__main__':
    unittest.main()
