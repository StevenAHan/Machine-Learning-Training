from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS:
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        # Initialize SBS with:
        # - estimator: ML model we want to use
        # - k_features: target number of features to select
        # - scoring: performance metric (default: accuracy)
        # - test_size: size of test split
        # - random_state: reproducibility
        self.scoring = scoring
        self.estimator = clone(estimator)  # clone model to keep original safe
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        # Split dataset into train/test sets
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
    
        # Start with all features
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))   # all feature indices
        self.subsets_ = [self.indices_]     # track feature subsets

        # Calculate initial score using all features
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        # Iteratively remove one feature at a time
        while dim > self.k_features:
            scores = []   # store scores for all feature combos
            subsets = []  # store feature subsets

            # Generate all combinations of features with (dim - 1)
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            # Pick the subset with the best score
            best = np.argmax(scores)
            self.indices_ = subsets[best]       # update current feature set
            self.subsets_.append(self.indices_) # store it
            dim -= 1                            # reduce dimension count

            # Store the best score at this stage
            self.scores_.append(scores[best])

        # Final best score with k_features
        self.k_score_ = self.scores_[-1]
        
        return self
        
    def transform(self, X):
        # Reduce dataset to selected feature subset
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # Train estimator on selected feature subset
        self.estimator.fit(X_train[:, indices], y_train)
        # Predict on test data
        y_pred = self.estimator.predict(X_test[:, indices])
        # Calculate and return score (e.g., accuracy)
        score = self.scoring(y_test, y_pred)
        return score
