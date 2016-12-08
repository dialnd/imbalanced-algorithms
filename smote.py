from collections import Counter
from random import choice

import numpy as np
from sklearn.neighbors import NearestNeighbors

class SMOTE(object):
    """Implementation of Synthetic Minority Over-Sampling Technique (SMOTE).

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors.

    Notes
    -----
    See the original paper for more details:
        [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE: 
            Synthetic Minority Over-Sampling Technique." Journal of Artificial 
            Intelligence Research (JAIR), 2002.

    Based on related code:
        - https://github.com/jeschkies/nyan/blob/master/shared_modules/smote.py
    """
    def __init__(self, k_neighbors, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def fit(self, X):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k+1)
        self.neigh.fit(self.X)

    def sample(self, N):
        """Generate samples.

        Parameters
        ----------
        N : int
            Number new synthetic samples.

        Returns
        -------
        S : array, shape = [(N/100) * n_minority_samples, n_features]
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(N, self.n_features))
        # Calculate synthetic samples.
        for i in range(N):
            j = np.random.randint(0, self.X.shape[0])

            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1), return_distance=False)[:, 1:]
            nn_index = choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

class BorderlineSMOTE(object):
    """Implementation of Borderline-SMOTE.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors.

    Notes
    -----
    See the original paper for more details:
        [1] H. Han, W-Y Wang, and B-H Mao. "Borderline-SMOTE: A New 
            Over-Sampling Method in Imbalanced Data Sets Learning." 
            International Conference on Intelligent Computing (ICIC), 2005.

    Based on related code:
        - https://github.com/jeschkies/nyan/blob/master/shared_modules/smote.py
    """
    def __init__(self, k_neighbors, return_mode='only', random_state=None):
        self.k = k_neighbors
        self.random_state = random_state
        self.return_mode = return_mode

    def fit(self, X, y):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n__samples, n_features]
            Holds the minority and majority samples.
        y : array-like, shape = [n__samples]
            Holds the class targets for samples.
        minority_target : int
            Value for minority class.
        """
        self.X = X
        self.y = y

        self.n_samples, _ = self.X.shape

        # Determine the minority class label.
        stats_c_ = Counter(y)
        maj_c_ = max(stats_c_, key=stats_c_.get)
        min_c_ = min(stats_c_, key=stats_c_.get)
        self.minority_target = min_c_

        # Learn nearest neighbors on complete training set.
        self.neigh = NearestNeighbors(n_neighbors=self.k+1)
        self.neigh.fit(self.X)

    def sample(self, N):
        """Generate samples.

        Parameters
        ----------
        N : int
            Number of new synthetic samples.

        Returns
        -------
        S : Synthetic samples of minorities in danger zone.
        safe : Safe minorities.
        danger : Minorities of danger zone.
        """
        np.random.seed(seed=self.random_state)

        safe_minority_indices = list()
        danger_minority_indices = list()

        for i in range(self.n_samples):
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
                # DANGER zone.
                danger_minority_indices.append(i)

        # SMOTE danger minority samples.
        smote = SMOTE1(self.k, random_state=self.random_state)
        smote.fit(self.X[danger_minority_indices])
        S = smote.sample(N)

        if self.return_mode == 'with_safe_and_danger':
            return (S, self.X[safe_minority_indices], self.X[danger_minority_indices])
        elif self.return_mode == 'only':
            return S
        else:
            pass