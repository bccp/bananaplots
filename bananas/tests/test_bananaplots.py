"""
Unit tests for Greeter class
"""
import unittest
import os
import bananas
import numpy
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm

from numpy.random import normal

def test_bananas():
    numpy.random.seed(1234)
    s1 = bananas.GMMSurface(
            X=numpy.concatenate([normal(size=10000), normal(size=10000)]),
            Y=numpy.concatenate([normal(size=10000), normal(size=10000)]),
            Z=numpy.concatenate([normal(size=10000), normal(size=10000)]),
        )
    s2 = bananas.GMMSurface(
            X=2 + normal(size=10000),
            Y=normal(size=10000), 
            Z=1 + normal(size=10000), 
            )

    banana = bananas.Bananas()

    banana.add_surface(s1, label="S1", cmap=cm.Reds_r)
    banana.add_surface(s2, label="S2", cmap=cm.Blues_r)
    banana.add_surface(s1 + s2, label="S1 + S2", cmap=cm.Greens_r, compiler_options=dict(nc=2))

    banana.set_feature("X", range=(-6, 6))
    banana.set_feature("Y", range=(-6, 6))
    banana.set_feature("Z", range=(-6, 6))

    fig = Figure()

    axes = banana.rendernd(fig, ["X", "Y", "Z"])

    handlers, labels = banana.get_legend_handlers_labels()
    axes[0, 2].legend(handlers, labels, loc='center')

    canvas = FigureCanvasAgg(fig)
    fig.tight_layout()
    fig.savefig("bananas-lower-left.png")

    fig = Figure()

    axes = banana.rendernd(fig, ["X", "Y", "Z"], corner='upper right')

    handlers, labels = banana.get_legend_handlers_labels()
    axes[2, 0].legend(handlers, labels, loc='center')

    canvas = FigureCanvasAgg(fig)
    fig.tight_layout()
    fig.savefig("bananas-upper-right.png")

if __name__ == '__main__':
    unittest.main()
