import numpy

__version__ = "0.0.3"

def _sorteditems(d, orderby):
    """ return items from a dict of dict, sorted by the orderby item of the dict """
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
            m = surface.compile((f1, f2), **compiler_options)
            Z = m.confidence(X, Y)

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

    def render1d(self, axes, f1, **options):
        axes.set_xlabel(self.features[f1]['label'])
        x = numpy.linspace(*self.features[f1]['range'], num=512)

        for surface, attrs in _sorteditems(self.surfaces, 'order'):
            compiler_options = attrs['compiler_options']
            m = surface.compile((f1, ), **compiler_options)
            Z = numpy.exp(m.lnprob(x))
            axes.plot(x, Z, label=attrs['label'], color=attrs['cmap'](0.3))

    def rendernd(self, figure, features, gridspec=None, **options):
        from matplotlib.gridspec import GridSpec
        from matplotlib.ticker import NullFormatter
        from itertools import product

        if gridspec is None:
            gridspec = GridSpec(len(features), len(features), hspace=0, wspace=0)

        corner = options.pop('corner', 'lower left')

        axes = {}

        config = {
                'upper right' : [lambda i, j : i < j, (0, 'top', len(features) - 1, 'right')],
                'lower left' :  [lambda i, j : i > j, (len(features) - 1, 'bottom', 0, 'left')]
                }

        for i, j in product(range(len(features)), range(len(features))):
            ax = figure.add_subplot(gridspec[i, j])
            axes[i, j] = ax
            visible = config[corner][0]
            if i == j:
                self.render1d(ax, features[i])
                ax.locator_params(axis='y', nbins=5)
                ax.yaxis.set_major_formatter(NullFormatter())
                continue

            if visible(i, j):
                self.render(ax, features[j], features[i])
            else:
                ax.set_axis_off()

        for (i, j), ax in axes.items():
            ax.locator_params(axis='y', prune='both')
            ax.locator_params(axis='x', prune='both')

        for (i, j), ax in axes.items():
            xedge, xpos, yedge, ypos = config[corner][1]
            if i != xedge:
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.xaxis.get_label().set_visible(False)
            else:
                ax.xaxis.set_label_position(xpos)
            if j != yedge:
                ax.yaxis.set_major_formatter(NullFormatter())
                ax.yaxis.get_label().set_visible(False)
            else:
                ax.yaxis.set_label_position(ypos)

        return axes

    def get_legend_handlers_labels(self):
        from matplotlib import patches as mpatches
        proxies = []
        labels = []
        for surface, attrs in _sorteditems(self.surfaces, 'order'):
            cmap = attrs['cmap']
            proxies.append(mpatches.Patch(color=cmap(0.3)))
            labels.append(attrs['label'])
        return proxies, labels

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

    def freeze(self, nc=1, nb=20, cov="full"):
        from itertools import product
        cache = {}
        # freeze 1d models
        for f1 in self.features:
            cache[(f1, )] = self.compile([f1], nc, nb, cov)

        # freeze 2d models
        for f1, f2 in product(self.features, self.features):
            cache[(f1, f2)] = self.compile([f1, f2], nc, nb, cov) 

        return FrozenSurface(cache, {})

    def compile(self, features, nc=1, nb=20, cov="full", ):
        """ compile the GMM to a function that returns
            ln_prob and confidence level as a function of features.

            We evaluate the log probability on all data points,
            then look for the percentiles. 

            Parameters
            ----------
            nb : number of bins in the interplation from ln_prob to CL
            nc : number of components to model the distribution.
            cov : type of covariance. 'full', 'diag', 'tied', 'spherical'

            Returns
            -------
            lnprob, confidence

        """
        data = []
        for feature in features:
            # only 1d feature is supported
            value = self.features[feature]
            data.append(value.reshape(1, -1))

        data = numpy.concatenate(data, axis=0)

        model = GMM(nc, covariance_type=cov)
        model.fit(data.T)
        lnprob = model.score(data.T)
        confidence_levels = 1 - numpy.logspace(-5, 0, num=nb)
        lnprob_cl = numpy.percentile(lnprob, 100 - confidence_levels * 100.)
        confidence_table = numpy.array([lnprob_cl, confidence_levels])

        return Marginalized(model, confidence_table)

class Marginalized(object):
    def __init__(self, model, confidence_table):
        self.model = model
        self.confidence_table = confidence_table 

    def lnprob(self, *args):
        args = numpy.array(numpy.broadcast_arrays(*args), copy=True)
        shape = args[0].shape
        args = args.reshape(len(args), -1)
        lnprob = self.model.score(args.T).reshape(shape)
        return lnprob

    def confidence(self, *args):
        x, y = self.confidence_table
        lnprob = self.lnprob(*args)
        return numpy.interp(lnprob, x, y)

class FrozenSurface(Surface):
    def __init__(self, cache, metadata):
        """
            Creates a picklable Frozen surface from a picklable model and a cl mapping.

            Parameters
            ----------
            clmapping : pairs of lnprob, confidence level.
            model : a model.

        """
        self.cache = cache
        self.metadata = metadata

    def __add__(self, other):
        raise TypeError("Cannot add two frozen surfaces")

    def freeze(self):
        return self

    def compile(self, features, **kwargs):
        return self.cache[tuple(features)]
