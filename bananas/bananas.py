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
        if not surface in self.surfaces:
            self.surfaces[surface] = dict(
                    colorfamily='r',
                    order=self._unique,
                    label=None,
                    cmap=None,
                    linewidth=1.0,
                    linestyle='-',
                    color=None,
                    levels=[0.68, 0.95],
                    compiler_options={},
                    )
            self._unique = self._unique + 10

        self.surfaces[surface].update(attrs)
        
        return surface

    def get_surface_attr(self, surface, attr):
        from matplotlib import cm

        f = self.surfaces[surface]
        if f[attr] is not None:
            return f[attr]

        if attr == 'label':
            return str(surface)

        if attr == 'color':
            cmap = self.get_surface_attr(surface, 'cmap')
            return cmap(0.3)

        if attr == 'cmap':
            color = f['colorfamily']
            shorts = {'b' : 'blue',
                         'r' : 'red',
                         'g' : 'green',
                         'y' : 'yellow',
                         'm' : 'magenta',
                         'k' : 'black'}
            color = shorts.get(color, color)

            return {'blue' : cm.Blues_r,
                        'red'  : cm.Reds_r,
                        'green'  : cm.Greens_r,
                        'yellow'  : cm.Oranges_r,
                        'magenta'  : cm.Purples_r,
                        'black'  : cm.Greys_r,
                        }[color]

    def set_feature(self, feature, **attrs):
        if not feature in self.features:
            self.features[feature] = dict(
                    order=self._unique,
                    label=None,
                    range=None,
                    )
            self._unique = self._unique + 10
        self.features[feature].update(attrs)

    def get_feature_attr(self, feature, attr):
        if not feature in self.features:
            self.set_feature(feature)

        f = self.features[feature]
        if f[attr] is not None:
            return f[attr]

        if attr == 'label':
            return str(feature)

        if attr == 'range':
            mins = [s[feature].vmin for s in self.surfaces]
            maxes = [s[feature].vmax for s in self.surfaces]
            return (min(mins), max(maxes))

    def render(self, axes, f1, f2, **options):
        axes.set_xlabel(self.get_feature_attr(f1, 'label'))
        axes.set_ylabel(self.get_feature_attr(f2, 'label'))

        x = numpy.linspace(*self.get_feature_attr(f1,'range'), num=512)
        y = numpy.linspace(*self.get_feature_attr(f2,'range'), num=512)
        X, Y = numpy.meshgrid(x, y)

        filled = options.get('filled', True)
        contour_labels = options.get('contour_labels', False)
        crosshair = options.get('crosshair', False)

        for surface, attrs in _sorteditems(self.surfaces, 'order'):
            cmap = self.get_surface_attr(surface, 'cmap')
            color = self.get_surface_attr(surface, 'color')
            compiler_options = self.get_surface_attr(surface, 'compiler_options')
            linestyle = self.get_surface_attr(surface, 'linestyle')
            linewidth = self.get_surface_attr(surface, 'linewidth')
            style = dict(linestyle=linestyle, linewidth=linewidth)

            levels = self.get_surface_attr(surface, 'levels')
            m = surface.compile((f1, f2), **compiler_options)
            Z = m.confidence(X, Y)
            if filled:
                CS = axes.contourf(X, Y, Z,
                        levels=[0] + levels,
                        vmin=0.0, vmax=1.0,
                        cmap=cmap, alpha=0.7)

            CS = axes.contour(X, Y, Z,
                    levels=levels,
                    vmin=0.0, vmax=2.0,
                    cmap=cmap, **style)

            if crosshair:
                x = surface[f1].peak
                y = surface[f2].peak
                if x is not None and y is not None:
                    axes.axvline(x, color=color, **style)
                    axes.axhline(y, color=color, **style)

            if contour_labels:
                TXT = axes.clabel(CS)

    def render1d(self, axes, f1, **options):
        crosshair = options.get('crosshair', False)
        range = self.get_feature_attr(f1, 'range')
        axes.set_xlabel(self.get_feature_attr(f1, 'label'))
        axes.set_xlim(range)
        x = numpy.linspace(*range, num=512)

        for surface, attrs in _sorteditems(self.surfaces, 'order'):
            label = self.get_surface_attr(surface, 'label')
            cmap = self.get_surface_attr(surface, 'cmap')
            color = self.get_surface_attr(surface, 'color')
            compiler_options = self.get_surface_attr(surface, 'compiler_options')
            linestyle = self.get_surface_attr(surface, 'linestyle')
            linewidth = self.get_surface_attr(surface, 'linewidth')
            style = dict(linestyle=linestyle, linewidth=linewidth)

            m = surface.compile((f1, ), **compiler_options)
            Z = numpy.exp(m.lnprob(x))
            axes.plot(x, Z, label=label, color=color, **style)

            if crosshair:
                c = surface[f1].peak
                if c is not None:
                    axes.axvline(c, color=color, **style)

    def rendernd(self, figure, features, gridspec=None, **options):
        from matplotlib.gridspec import GridSpec
        from matplotlib.ticker import NullFormatter
        from itertools import product

        if gridspec is None:
            gridspec = GridSpec(len(features), len(features), hspace=0, wspace=0)

        corner = options.get('corner', 'lower left')

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
                self.render1d(ax, features[i], **options)
                ax.locator_params(axis='y', nbins=5)
                ax.yaxis.set_major_formatter(NullFormatter())
                continue

            if visible(i, j):
                self.render(ax, features[j], features[i], **options)
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
            label = self.get_surface_attr(surface, 'label')
            color = self.get_surface_attr(surface, 'color')
            proxies.append(mpatches.Patch(color=color))
            labels.append(label)
        return proxies, labels

class Surface(object):
    def __getitem__(self, name):
        return self.features[name]

    pass

class Feature(object):
    def __init__(self, data, vmin=None, vmax=None, peak=None):
        if isinstance(data, Feature):
            if vmin is None:
                vmin = data.vmin
            if vmax is None:
                vmax = data.vmax
            if peak is None:
                peak = data.peak
            data = data.data
        else:
            if vmin is None:
                vmin = data.min()
            if vmax is None:
                vmax = data.max()

        # only 1d feature is supported
        assert len(numpy.shape(data)) == 1
        self.data = data
        self.vmin = vmin
        self.vmax = vmax
        self.peak = peak

    def __add__(self, other):
        return Feature(numpy.concatenate([self.data, other.data]),
                    vmin=numpy.min([self.vmin, other.vmin]),
                    vmax=numpy.max([self.vmax, other.vmax]),
                    peak=None)

class GMMSurface(Surface):
    """
        A log-likelyhood surface generated from Gausian mixture.

    """
    def __init__(self, **features):

        self.features = {}
        for name, feature in features.items():
            self.features[name] = Feature(feature)

    def __add__(self, other):
        if not isinstance(other, GMMSurface):
            return Surface.__add__(self, other)

        features = {}
        for name in self.features:
            if not name in other.features:
                continue
            features[name] = self.features[name] + other.features[name]
        return GMMSurface(**features)

    def freeze(self, nc=1, nb=20, cov="full"):
        from itertools import product
        cache = {}
        features = {}
        # freeze features
        for f1 in self.features:
            feature = self.features[f1]
            features[f1] = Feature([], feature.vmin, feature.vmax, feature.peak)

        # freeze 1d models
        for f1 in self.features:
            cache[(f1, )] = self.compile([f1], nc, nb, cov)

        # freeze 2d models
        for f1, f2 in product(self.features, self.features):
            if f1 == f2 : continue
            if (f1, f2) in cache: continue
            m = self.compile([f1, f2], nc, nb, cov) 
            cache[(f1, f2)] = m
            cache[(f2, f1)] = m

        return FrozenSurface(features, cache, dict(nc=nc, nb=nb, cov=cov))

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
        limits = []
        for name in features:
            # only 1d name is supported
            feature = self.features[name]
            data.append(feature.data.reshape(1, -1))
            limits.append((feature.vmin, feature.vmax))
        data = numpy.concatenate(data, axis=0)
        limits = numpy.array(limits)

        from sklearn import mixture

        # XXX: Do not use DPGMM because the normalization is buggy
        # https://github.com/scikit-learn/scikit-learn/issues/7371
        model = mixture.GMM(nc, covariance_type=cov, n_iter=100)
        model.fit(data.T)

        if not model.converged_:
            raise ValueError("Your data is strange. Gaussian mixture failed to converge")

        lnprob = model.score(data.T)
        confidence_levels = 1 - numpy.logspace(-5, 0, num=nb)
        lnprob_cl = numpy.percentile(lnprob, 100 - confidence_levels * 100.)
        confidence_table = numpy.array([lnprob_cl, confidence_levels])

        return Marginalized(model, confidence_table, limits)

class Marginalized(object):
    def __init__(self, model, confidence_table, limits):
        self.model = model
        self.confidence_table = confidence_table 
        self.mins  = limits[:, 0]
        self.maxes = limits[:, 1]
    def lnprob(self, *args):
        args = numpy.array(numpy.broadcast_arrays(*args), copy=True)
        shape = args[0].shape
        args = args.reshape(len(args), -1)
        mask = (args >= self.mins[:, None]).all(axis=0)
        mask &= (args <= self.maxes[:, None]).all(axis=0)
        lnprob = self.model.score(args.T)
        lnprob[~mask] = - numpy.inf
        lnprob = lnprob.reshape(shape)
        return lnprob

    def confidence(self, *args):
        x, y = self.confidence_table
        lnprob = self.lnprob(*args)
        return numpy.interp(lnprob, x, y, left=1., right=0.)

class FrozenSurface(Surface):
    def __init__(self, features, cache, metadata):
        """
            Creates a picklable Frozen surface from a picklable model and a cl mapping.

            Parameters ----------
            clmapping : pairs of lnprob, confidence level.
            model : a model.

        """
        self.cache = cache
        self.metadata = metadata
        self.features = features

    def __add__(self, other):
        raise TypeError("Cannot add two frozen surfaces")

    def freeze(self):
        return self

    def compile(self, features, **kwargs):
        return self.cache[tuple(features)]
