import numpy
# FIXME: copy the functions here

from sklearn.mixture.gmm import log_multivariate_normal_density, logsumexp, sample_gaussian

class GMM(object):
    def __init__(self, weights, means, covs):
        self.weights = numpy.array(weights)
        self.means = numpy.array(means)
        self.covs = numpy.array(covs)

    def score(self, X, return_responsibilities=False):
        nc = len(self.weights)
        X = numpy.array(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.size == 0:
            return numpy.array([]), numpy.empty((0, len(self.weights)))

        if X.shape[1] != self.means.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

        lpr = numpy.log(self.weights) + \
              log_multivariate_normal_density(X,
                    self.means,
                    self.covs, 'full')

        logprob = logsumexp(lpr, axis=1)
        if return_responsibilities:
            responsibilities = numpy.exp(lpr - logprob[:, None])
            return logprob, responsibilities
        return logprob

    def marginalize(self, axes):
        return GMM(self.weights, self.means[..., axes], self.covs[..., axes][..., axes, :])

    def sample(self, size, random_state=None):
        """Generate random samples from the model.
        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples
        """

        if random_state is None:
            random_state = numpy.random


        X = numpy.empty(size, ('f8', (self.means.shape[1],)))
        # decide which component to use for each sample
        comps = random_state.choice(len(self.weights), p=self.weights, size=size)
        # for each component, generate all needed samples
        for comp in range(len(self.weights)):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                cv = self.covs[comp]
                g = sample_gaussian(
                    self.means[comp], cv, 'full',
                    num_comp_in_X, random_state=random_state).T
                X[comp_in_X] = g
        return X

    @classmethod
    def fit(kls, nc, X):
        # FIXME: get rid of this and add weights support
        from sklearn import mixture

        # XXX: Do not use DPGMM because the normalization is buggy
        # https://github.com/scikit-learn/scikit-learn/issues/7371

        model = mixture.GMM(nc, covariance_type='full', n_iter=100)
        model.fit(X)

        if not model.converged_:
            raise ValueError("Your data is strange. Gaussian mixture failed to converge")

        return kls(model.weights_, model.means_, model.covars_)

class Confidence(object):
    def __init__(self, model, confidence_table):
        self.model = model
        self.confidence_table = confidence_table
        
    def score(self, sc):
        x, y = self.confidence_table
        return numpy.interp(sc, x, y, left=1., right=0.)

    @classmethod
    def fit(kls, model, nsample=16*1024, vmin=-5, vmax=0, nb=100):
        X = model.sample(nsample)
        sc = model.score(X)
        confidence_levels = 1 - numpy.logspace(vmin, vmax, num=nb)
        # FIXME: add weight support here
        sc_cl = numpy.percentile(sc, 100. - confidence_levels * 100.)
        confidence_table = numpy.array([sc_cl, confidence_levels])
        return kls(model, confidence_table)

