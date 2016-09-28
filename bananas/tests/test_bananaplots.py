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
    data = normal(size=(3, 2000))
    s1 = bananas.MCSurface(
            X=bananas.Feature(data[0], vmin=-5, vmax=5, peak=0.0),
            Y=bananas.Feature(data[1], vmin=-5, vmax=5, peak=0.0),
            Z=bananas.Feature(data[2], vmin=-5, vmax=5, peak=0.0),
        )
    s2 = bananas.MCSurface(
            X=bananas.Feature(1 + data[0], vmin=-5, vmax=5, peak=1.0),
            Y=bananas.Feature(2 + data[1], vmin=-5, vmax=5, peak=2.0),
            Z=bananas.Feature(3 + data[2], vmin=-5, vmax=5, peak=3.0),
            )

    s3 = (s1 + s2).freeze()
    banana = bananas.Bananas()

    banana.add_surface(s1, label="S1", colorfamily='r')
    banana.add_surface(s2, label="S2", colorfamily='b')
    banana.add_surface(s3, label="S1 + S2", colorfamily='g', linestyle='--', compiler_options=dict(nc=2))

#    banana.set_feature("X", range=(-6, 6))
#    banana.set_feature("Y", range=(-3, 4))
#    banana.set_feature("Z", range=(-6, 4))

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

def test_freeze():
    numpy.random.seed(1234)
    s1 = bananas.MCSurface(
            X=numpy.concatenate([normal(size=1000), normal(size=1000)]),
            Y=numpy.concatenate([normal(size=1000), normal(size=1000)]),
            Z=numpy.concatenate([normal(size=1000), normal(size=1000)]),
        )
    s2 = bananas.MCSurface(
            X=2 + normal(size=1000),
            Y=normal(size=1000), 
            Z=1 + normal(size=1000), 
            )

    import pickle
    f2 = s2.freeze(nc=20, nb=100)
    s = pickle.dumps(f2)
    print(len(s))
    f2 = pickle.loads(s)

def test_1d():
    """ Overall integration test """

    numpy.random.seed(1234)
    s1 = bananas.MCSurface(
            X=numpy.concatenate([normal(size=2000)]),
        )
    #numpy.random.seed(1234)
    s2 = bananas.MCSurface(
            X=0 + normal(size=3000),
            )

    #s3 = (s1 + s2).freeze()
    banana = bananas.Bananas()

    banana.add_surface(s1, label="S1", cmap=cm.Reds_r)
    banana.add_surface(s2, label="S2", cmap=cm.Blues_r)
    #banana.add_surface(s3, label="S1 + S2", cmap=cm.Greens_r, compiler_options=dict(nc=2))

    banana.set_feature("X", range=(-6, 6))
    banana.set_feature("Y", range=(-3, 3))
    banana.set_feature("Z", range=(-6, 6))

    fig = Figure()

    axes = fig.add_subplot(111)

    banana.render1d(axes, "X", nc=1, nb=100)

    handlers, labels = banana.get_legend_handlers_labels()
    axes.legend(handlers, labels, loc='center')

    canvas = FigureCanvasAgg(fig)
    fig.tight_layout()
    fig.savefig("bananas-1d.png")

if __name__ == '__main__':
    unittest.main()
