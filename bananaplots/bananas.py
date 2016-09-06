from sklearn.mixture import GMM
from matplotlib.colors import LogNorm
from matplotlib import cm
import numpy

def _sorteditems(d, orderby):
    s = sorted([(i[orderby], k) for k, i in d.items()])
    return [(k, d[k]) for i, k in s]

class Bananas(object):
    def __init__(self):
        self.features = {}
        self._unique = 0
        self.surfaces = {}

    def add_surface(self, surface, **attrs):
        """
            Add a surface with attributes.
            
            Notes
            -----
                compiler attributes are prefixed with 'compiler_'
            Returns
            -------
                the surface object
        """
        if not surface in self.surfaces:
            self.surfaces[surface] = dict(
                    order=self._unique,
                    label=str(surface),
                    cmap=cm.jet,
                    compiler={},
                    )
            self._unique = self._unique + 10
        self.surfaces[surface].update(attrs)

        return surface

    def set_feature(self, feature, **attrs):
        if not feature in self.features:
            self.features[feature] = dict(
                    order=self._unique,
                    label=str(feature),
                    range=(-1., 1.)
                    )
            self._unique = self._unique + 10
        self.features[feature].update(attrs)

    def render(self, axes, f1, f2):
        axes.set_xlabel(self.features[f1]['label'])
        axes.set_ylabel(self.features[f2]['label'])
        x = numpy.linspace(*self.features[f1]['range'])
        y = numpy.linspace(*self.features[f2]['range'])
        X, Y = numpy.meshgrid(x, y)
        for surface, attrs in _sorteditems(self.surfaces, 'order'):
            compiler_attrs = attrs['compiler']
            func = surface.compile((f1, f2), **compiler_attrs)
            Z = func(X, Y)
            CS = axes.contour(X, Y, Z,
                    levels=[0, 4, 8],
                    cmap=attrs['cmap'],
                    label=attrs['label'], alpha=1.0)

class Surface(object):
    def __add__(self, other):
        return CombinedSurface(self, other)

class GMMSurface(Surface):
    """
        A log-likelyhood surface generated from Gausian mixture.

    """
    def __init__(self, **features):

        self.features = {}
        for feature, value in features.items():
            # only 1d feature is supported
            assert len(numpy.shape(value)) == 1
            self.features[feature] = value

    def __add__(self, other):
        if not isinstance(other, GMMSurface):
            return Surface.__add__(self, other)

        features = {}
        for feature in self.features:
            if not feature in other.features:
                continue
            features[feature] = numpy.concatenate(
                    [self.features[feature], other.features[feature]])
        return GMMSurface(**features)

    def compile(self, features, nc=1):
        data = []
        for feature in features:
            # only 1d feature is supported
            value = self.features[feature]
            data.append(value.reshape(1, -1))

        data = numpy.concatenate(data, axis=0)

        model = GMM(nc)
        model.fit(data.T)
        def func(*args):
            args = numpy.array(numpy.broadcast_arrays(*args), copy=True)
            shape = args[0].shape
            args = args.reshape(len(args), -1)
            # FIXME: returns confidence level instead!
            return -model.score(args.T).reshape(shape)
        return func

class CombinedSurface(Surface):
    def __init__(self, *surfaces):
        raise RuntimeError("I don't think this is correct. How to pass in the covariance")
        s = []
        # apply the association rule
        # but do not go deeper for nested.
        for surface in surfaces:
            print(surface)
            if isinstance(surface, CombinedSurface):
                s.extend(surface.surfaces)
            else:
                s.append(surface)
        self.surfaces = s
        print(self.surfaces)

    def compile(self, features, **kwargs):
        frozen = []
        for s in self.surfaces:
            frozen.append(s.compile(features, **kwargs))
        def func(*args):
            s = []
            for f in frozen:
                s.append(f(*args))
            return sum(s)
        return func
