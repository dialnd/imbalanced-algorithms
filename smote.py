from collections import Counter
from random import choice

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import shuffle

class SMOTE(object):
    """Implementation of Synthetic Minority Over-Sampling Technique (SMOTE).

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors.

    References
    ----------
        [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE: 
            Synthetic Minority Over-Sampling Technique." Journal of Artificial 
            Intelligence Research (JAIR), 2002.
    """
    def __init__(self, k_neighbors=3, return_mode='only', random_state=None):
        self.k = k_neighbors
        self.return_mode = return_mode
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
            j = np.random.randint(0, self.X.shape[0])

            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1), return_distance=False)[:, 1:]
            nn_index = choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        if self.return_mode == 'append':
            X = np.vstack((self.X, S))
            y = np.full(X.shape[0], fill_value=self.minority_target)
            return (X, y)
        elif self.return_mode == 'only':
            return S
        else:
            pass

    def fit(self, X, minority_target=1):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        minority_target : int
            Minority class label.
        """
        self.X = X
        self.minority_target = minority_target
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k+1)
        self.neigh.fit(self.X)

        return self

class BorderlineSMOTE(object):
    """Implementation of Borderline-SMOTE.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors.

    References
    ----------
        [1] H. Han, W-Y Wang, and B-H Mao. "Borderline-SMOTE: A New 
            Over-Sampling Method in Imbalanced Data Sets Learning." 
            International Conference on Intelligent Computing (ICIC), 2005.
    """
    def __init__(self, k_neighbors=3, return_mode='only', random_state=None):
        self.k = k_neighbors
        self.return_mode = return_mode
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.

        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.

        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Synthetic samples of minority samples in the danger zone.
        safe : array, shape = [safe_minority_samples, n_features]
            Safe minority samples.
        danger : array, shape = [danger_minority_samples, n_features]
            Minority samples in the danger zone.
        """
        np.random.seed(seed=self.random_state)

        safe_minority_indices = list()
        danger_minority_indices = list()

        for i in range(self.sample_count):
            if self.y[i] != self.minority_target:
                continue

            nn = self.neigh.kneighbors(self.X[i].reshape(1, -1), return_distance=False)[:, 1:]

            majority_neighbors = 0
            for n in nn[0]:
                if self.y[n] != self.minority_target:
                    majority_neighbors += 1

            if majority_neighbors == len(nn[0]):
                continue
            elif majority_neighbors < (len(nn[0]) / 2):
                # Add sample to safe minorities.
                safe_minority_indices.append(i)
            else:
                # Danger zone.
                danger_minority_indices.append(i)

        # SMOTE danger minority samples.
        smote = SMOTE(self.k, random_state=self.random_state)
        smote.fit(self.X[danger_minority_indices])
        S = smote.sample(n_samples)

        if self.return_mode == 'with_safe_and_danger':
            return (S, self.X[safe_minority_indices], self.X[danger_minority_indices])
        elif self.return_mode == 'append':
            X = np.vstack((self.X, S))
            y = np.full(X.shape[0], fill_value=self.minority_target)
            return (X, y)
        elif self.return_mode == 'only':
            return S
        else:
            pass

    def fit(self, X, y, minority_target=None):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n__samples, n_features]
            Holds the minority and majority samples.
        y : array-like, shape = [n__samples]
            Holds the class targets for samples.
        minority_target : int
            Minority class label.
        
        Returns
        -------
        self : object
            Returns self.
        """
        self.X = X
        self.y = y

        self.sample_count, _ = self.X.shape

        if minority_target:
            self.minority_target = minority_target
        else:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_

        # Learn nearest neighbors on complete training set.
        self.neigh = NearestNeighbors(n_neighbors=self.k+1)
        self.neigh.fit(self.X)

        return self

class SMOTEBoost(AdaBoostClassifier):
    """Implementation of SMOTEBoost.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors.
    n_samples : int
        Number of new synthetic samples per boosting step.

    References
    ----------
        [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer. 
            "SMOTEBoost: Improving Prediction of the Minority Class in 
            Boosting." European Conference on Principles of Data Mining and 
            Knowledge Discovery (PKDD), 2003.
    """
    def __init__(self,
                 k_neighbors=3,
                 n_samples=100,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.algorithm = algorithm

        self.smote = SMOTE(k_neighbors=k_neighbors, return_mode='only')
        self.n_samples = n_samples

        super(SMOTEBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing SMOTE during each boosting step.

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
        Based on the scikit-learn v0.18 AdaBoostClassifier `fit` method.
        """
        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64 # from fast_dict.pxd
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
            # SMOTE step.
            X_min = X[np.where(y == self.minority_target)]
            self.smote.fit(X_min)
            X_syn = self.smote.sample(self.n_samples)
            y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target, dtype=np.int64)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the original and synthetic instances.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            # Combine the weights.
            sample_weight = np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            X, y, sample_weight = shuffle(X, y, sample_weight)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

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

    def predict(self, X):
        return super(SMOTEBoost, self).predict(X)
