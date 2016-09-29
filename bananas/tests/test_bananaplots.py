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
    """ Overall integration test """

    numpy.random.seed(1234)
    data = normal(size=(3, 10000))
    data[2][data[2] < 0] += 3

    c1 = bananas.MCChain(
            X=bananas.Feature(data[0], peak=0.0),
            Y=bananas.Feature(data[1], peak=0.0),
            Z=bananas.Feature(data[2], peak=0.0),
        )
    c2 = bananas.MCChain(
            X=bananas.Feature(1 + data[0], peak=1.0),
            Y=bananas.Feature(2 + data[1], peak=2.0),
            Z=bananas.Feature(3 + data[2], peak=3.0),
            )

    c3 = (c1 + c2)
    banana = bananas.Bananas()
    s1 = c1.compile(nc=10)
    s2 = c2.compile(nc=10)
    s3 = s1 * s2

    banana.set_surface(s1, label="S1", colorfamily='r')
    banana.set_surface(s2, label="S2", colorfamily='b')
    banana.set_surface(s3, label="S1 + S2", colorfamily='g', linestyle='--')

#    banana.set_feature("X", range=(-6, 6))
#    banana.set_feature("Y", range=(-3, 4))
    banana.set_feature("Z", range=(-4, 4))

    fig = Figure()

    axes = banana.rendernd(fig, ["X", "Y", "Z"], nc=1, nb=100,
            contour_labels=True,
            filled=True,
            crosshair=True)

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

    fig = Figure()

    axes = fig.add_subplot(111)

    banana.render1d(axes, "X")

    handlers, labels = banana.get_legend_handlers_labels()
    axes.legend(handlers, labels, loc='center')

    canvas = FigureCanvasAgg(fig)
    fig.tight_layout()
    fig.savefig("bananas-1d.png")

def test_freeze():
    numpy.random.seed(1234)
    s1 = bananas.MCChain(
            X=numpy.concatenate([normal(size=1000), normal(size=1000)]),
            Y=numpy.concatenate([normal(size=1000), normal(size=1000)]),
            Z=numpy.concatenate([normal(size=1000), normal(size=1000)]),
        )
    s2 = bananas.MCChain(
            X=2 + normal(size=1000),
            Y=normal(size=1000), 
            Z=1 + normal(size=1000), 
            )

    import pickle
    f2 = s2.compile(nc=20, nb=100)
    s = pickle.dumps(f2)
    print(len(s))
    f2 = pickle.loads(s)

if __name__ == '__main__':
    unittest.main()
