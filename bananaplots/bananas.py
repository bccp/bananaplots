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
        from matplotlib import cm

        if not surface in self.surfaces:
            self.surfaces[surface] = dict(
                    order=self._unique,
                    label=str(surface),
                    cmap=cm.jet,
                    compiler_options={},
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

    def render(self, axes, f1, f2, **options):
        axes.set_xlabel(self.features[f1]['label'])
        axes.set_ylabel(self.features[f2]['label'])

        x = numpy.linspace(*self.features[f1]['range'], num=512)
        y = numpy.linspace(*self.features[f2]['range'], num=512)
        X, Y = numpy.meshgrid(x, y)

        for surface, attrs in _sorteditems(self.surfaces, 'order'):
            compiler_options = attrs['compiler_options']
            func = surface.compile((f1, f2), **compiler_options)
            Z = func(X, Y)

            if options.pop('filled', True):
                CS = axes.contourf(X, Y, Z,
                        levels=[0, 0.68, 0.95],
                        vmin=0.0, vmax=1.0,
                        cmap=attrs['cmap'], alpha=0.7)

            CS = axes.contour(X, Y, Z,
                    levels=[0.68, 0.95],
                    vmin=0.0, vmax=2.0,
                    cmap=attrs['cmap'])

            if options.pop('contour_labels', False):
                TXT = axes.clabel(CS)

    def get_legend_handlers_labels(self):
        from matplotlib import patches as mpatches
        proxies = []
        labels = []
        for surface, attrs in _sorteditems(self.surfaces, 'order'):
            cmap = attrs['cmap']
            proxies.append(mpatches.Patch(color=cmap(0.6)))
            labels.append(attrs['label'])
        return proxies, labels

from scipy.optimize import minimize
def findcl(func, X, Y, levels):
    XC = (X[1:, 1:] + X[:-1, :-1]) * 0.5
    YC = (Y[1:, 1:] + Y[:-1, :-1]) * 0.5
    A = abs((X[1:, 1:] - X[:-1, :-1]) * (Y[1:, 1:] - Y[:-1, :-1]))
    ZC = func(XC, YC)
    ZA = ZC * A
    r = []
    last = ZC.max()
    for level in levels:
        if level >= 1.0:
            r.append(0)
            continue
        if level <= 0.0:
            r.append(ZC.max())
            continue

        def cost(thresh):
            mask = ZC > thresh[0]
            cost = (ZA[mask].sum() - level) ** 2
            return cost
        res = minimize(cost, (last, ), method='Nelder-Mead')
        r.append(res.x[0])
    return numpy.array(r)

class Surface(object):
    def __add__(self, other):
        return CombinedSurface(self, other)

from sklearn.mixture import GMM
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

    def compile(self, features, nc=1, nb=20):
        """ compile the GMM to a function that returns
            confidence level as a function of features.

            We evaluate the log probability on all data points,
            then look for the percentiles. 

            Parameters
            ----------
            nb : number of bins in the interplation from ln_prob to CL
            nc : number of components to model the distribution.

        """
        data = []
        for feature in features:
            # only 1d feature is supported
            value = self.features[feature]
            data.append(value.reshape(1, -1))

        data = numpy.concatenate(data, axis=0)

        model = GMM(nc)
        model.fit(data.T)
        lnprob = model.score(data.T)
        confidence_levels = 1 - numpy.logspace(-5, 0, num=nb)
        lnprob_cl = numpy.percentile(lnprob, 100 - confidence_levels * 100.)

        def func(*args):
            args = numpy.array(numpy.broadcast_arrays(*args), copy=True)
            shape = args[0].shape
            args = args.reshape(len(args), -1)
            lnprob = model.score(args.T).reshape(shape)
            return numpy.interp(lnprob, lnprob_cl, confidence_levels)

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
        raise
