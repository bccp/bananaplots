import numpy

__version__ = "0.0.3"

from .model import GMM, Confidence

def _sorteditems(d, orderby):
    """ return items from a dict of dict, sorted by the orderby item of the dict """
    s = sorted([(i[orderby], k) for k, i in d.items()])
    return [(k, d[k]) for i, k in s]

class Bananas(object):
    def __init__(self):
        self.features = {}
        self._unique = 0
        self.surfaces = {}

    def set_surface(self, surface, **attrs):
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
            linestyle = self.get_surface_attr(surface, 'linestyle')
            linewidth = self.get_surface_attr(surface, 'linewidth')
            style = dict(linestyle=linestyle, linewidth=linewidth)

            levels = self.get_surface_attr(surface, 'levels')
            m = surface.marginalize((f1, f2))
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
            linestyle = self.get_surface_attr(surface, 'linestyle')
            linewidth = self.get_surface_attr(surface, 'linewidth')
            style = dict(linestyle=linestyle, linewidth=linewidth)

            m = surface.marginalize((f1, )) 
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

    def marginalize(self, features, **options):
        axes = []
        for name in features:
            axes.append(self.names.index(name))

        model = self.model.marginalize(axes)
        conf = Confidence.fit(model, **options)
        return Marginalized(model, conf)
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

class MCChain(Surface):
    """
        A log-likelyhood surface represented by a Markov Chain sample.

        Parameters
        ----------
        **features : dict
            key: name of the feature,
            value : a :py:class:`Feature` object or a 1-d numpy array.
                    array will be cast to a :py:class:`Feature` object.

    """
    def __init__(self, **features):
        self.features = {}
        for name, feature in features.items():
            self.features[name] = Feature(feature)

    def __add__(self, other):
        features = {}
        for name in self.features:
            if not name in other.features:
                continue
            features[name] = self.features[name] + other.features[name]
        return MCChain(**features)

    def compile(chain, nc=1, nb=20):
        data = []
        names = []
        limits = []
        for name in chain.features:
            # only 1d name is supported
            feature = chain.features[name]
            data.append(feature.data.reshape(1, -1))
            # remove the data from feature
            names.append((name, Feature([], feature.vmin, feature.vmax, feature.peak)))
            limits.append((feature.vmin, feature.vmax))

        X = numpy.concatenate(data, axis=0).T

        model = GMM.fit(nc, X, limits)
        conf = Confidence.fit(model, nb=nb)

        return GMMSurface(names, model)


class Marginalized(object):
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf 

    def lnprob(self, *args):
        args = numpy.array(numpy.broadcast_arrays(*args), copy=True)
        shape = args[0].shape
        args = args.reshape(len(args), -1)
        X = args.T
        lnprob = self.model.score(X)
        lnprob = lnprob.reshape(shape)
        return lnprob

    def confidence(self, *args):
        lnprob = self.lnprob(*args)
        return self.conf.score(lnprob)

class CombinedSurface(Surface):
    def __init__(self, surfaces):
        names = []
        for s in surfaces:
            names.extend(s.names)
        common = list(set(names))

        axes = []
        for s in surfaces:
            axes.append([ s.names.index(name) for name in common])

        features = []
        for name in common:
            f = reduce(lambda x, y : x + y, [s.features[name] for s in surfaces])
            features.append((name, f))

        self.features = dict(features)
        self.names = common

        models = [surface.model.marginalize(axes0) for surface, axes0 in zip(surfaces, axes)]
        self.model = CombinedModel(models)

    def marginalize(self, features, **options):
        axes = []
        for name in features:
            axes.append(self.names.index(name))

        model = self.model.marginalize(axes)
        conf = Confidence.fit(model, **options)
        return Marginalized(model, conf)

class CombinedModel(object):
    def __init__(self, models):
        self.models = models

    def score(self, X):
        return sum([model.score(X) for model in self.models])

    def marginalize(self, axes):
        return CombinedModel([
            model.marginalize(axes) for model in self.models])

    def sample(self, nsample, random_state=None):
        if random_state is None:
            random_state = numpy.random

        def once(size):
            X = self.models[0].sample(size, random_state)
            nf = X.shape[-1]
            lnprob = sum([model.score(X) for model in self.models[1:]])
            prob = numpy.exp(lnprob)
            prob /= prob.max()
            keep = random_state.rand(len(X)) < prob
            return X[keep].reshape(-1, nf)
        g = once(nsample)
        ng = nsample
        while len(g) < nsample:
            togen = (nsample - len(g)) * ng // len(g)
            g1 = once(togen)
            ng = ng + togen
            g = numpy.append(g, g1, axis=0)
        return g[:nsample]

class GMMSurface(Surface):
    """ A surface that is modelled by GMM. features is a list of (name, feature). """

    def __init__(self, features, model):
        self.features = dict(features)
        self.names = [feature[0] for feature in features]
        self.model = model

    def __mul__(self, other):
        return CombinedSurface([self, other])

