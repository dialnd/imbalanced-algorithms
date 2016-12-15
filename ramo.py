from collections import Counter

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
#from sklearn.utils import shuffle


class RankedMinorityOversampler(object):
    """Implementation of Ranked Minority Oversampling (RAMO).

    Oversample the minority class by picking samples according to a specified
    sampling distribution.

    Parameters
    ----------
    k_neighbors_1 : int, optional (default=5)
        Number of nearest neighbors used to adjust the sampling probability of
        the minority examples.
    k_neighbors_2 : int, optional (default=5)
        Number of nearest neighbors used to generate the synthetic data
        instances.
    alpha : float, optional (default=0.3)
        Scaling coefficient.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self, k_neighbors_1=5, k_neighbors_2=5, alpha=0.3,
                 random_state=None):
        self.k_neighbors_1 = k_neighbors_1
        self.k_neighbors_2 = k_neighbors_2
        self.alpha = alpha
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.

        Parameters
        ----------
        n_samples : int
            Number new synthetic samples.

        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            # Choose a sample according to the sampling distribution, r.
            j = np.random.choice(self.n_minority_samples, p=self.r)

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh_2.kneighbors(self.X_min[j].reshape(1, -1),
                                         return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X_min[nn_index] - self.X_min[j]
            gap = np.random.random()

            S[i, :] = self.X_min[j, :] + gap * dif[:]

        return S

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_total_samples, n_features]
            Holds the majority and minority samples.
        y : array-like, shape = [n_total_samples]
            Holds the class targets for samples.
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights multiplier. If None, the multiplier is 1.
        minority_target : int, optional (default=None)
            Minority class label.
        """
        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        self.X_min = X[y == self.minority_target]
        self.n_minority_samples, self.n_features = self.X_min.shape

        neigh_1 = NearestNeighbors(n_neighbors=self.k_neighbors_1 + 1)
        neigh_1.fit(X)
        nn = neigh_1.kneighbors(self.X_min, return_distance=False)[:, 1:]

        if sample_weight is None:
            sample_weight_min = np.ones(shape=(len(self.minority_target)))
        else:
            assert(len(y) == len(sample_weight))
            sample_weight_min = sample_weight[y == self.minority_target]

        self.r = np.zeros(shape=(self.n_minority_samples))
        for i in range(self.n_minority_samples):
            majority_neighbors = 0
            for n in nn[i]:
                if y[n] != self.minority_target:
                    majority_neighbors += 1

            self.r[i] = 1. / (1 + np.exp(-self.alpha * majority_neighbors))

        self.r = (self.r * sample_weight_min).reshape(1, -1)
        self.r = np.squeeze(normalize(self.r, axis=1, norm='l1'))

        # Learn nearest neighbors.
        self.neigh_2 = NearestNeighbors(n_neighbors=self.k_neighbors_2 + 1)
        self.neigh_2.fit(self.X_min)

        return self


class RAMOBoost(AdaBoostClassifier):
    """Implementation of RAMOBoost.

    RAMOBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class according to a specified sampling
    distribution on each boosting iteration [1].

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    k_neighbors_1 : int, optional (default=5)
        Number of nearest neighbors used to adjust the sampling probability of
        the minority examples.
    k_neighbors_2 : int, optional (default=5)
        Number of nearest neighbors used to generate the synthetic data
        instances.
    alpha : float, optional (default=0.3)
        Scaling coefficient.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    References
    ----------
    .. [1] S. Chen, H. He, and E. A. Garcia. "RAMOBoost: Ranked Minority
           Oversampling in Boosting". IEEE Transactions on Neural Networks,
           2010.
    """

    def __init__(self,
                 n_samples=100,
                 k_neighbors_1=5,
                 k_neighbors_2=5,
                 alpha=0.3,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.ramo = RankedMinorityOversampler(k_neighbors_1, k_neighbors_2,
                                              alpha, random_state=random_state)

        super(RAMOBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # RAMO step.
            self.ramo.fit(X, y, sample_weight=sample_weight)
            X_syn = self.ramo.sample(self.n_samples)
            y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                            dtype=np.int64)

            # Combine the minority and majority class samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            #print('sw', sample_weight)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self
