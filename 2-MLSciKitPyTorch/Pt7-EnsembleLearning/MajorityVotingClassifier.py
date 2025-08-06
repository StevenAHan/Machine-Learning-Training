from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
import numpy as np


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        """
        Custom Majority Vote Classifier.
        
        Parameters
        ----------
        classifiers : list
            A list of scikit-learn classifiers.
        
        vote : str, default='classlabel'
            Voting strategy: 
            - 'classlabel': majority vote on predicted labels
            - 'probability': average predicted probabilities
        
        weights : list or None
            Optional weight for each classifier. 
            Higher weight = more influence in voting.
        """
        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """Fit classifiers on the training data."""
        # ensure correct voting type
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote must be 'probability' "
                             f"or 'classlabel'; got (vote={self.vote})")

        # ensure weights length matches classifiers
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and weights must be equal; '
                             f'got {len(self.weights)} weights, '
                             f'{len(self.classifiers)} classifiers')

        # encode labels to integers starting from 0
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_

        # clone each classifier to keep them independent
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """Predict class labels based on voting."""
        if self.vote == 'probability':
            # pick class with highest average probability
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' voting
            # collect predictions from all classifiers
            predictions = np.asarray([
                clf.predict(X) for clf in self.classifiers_
            ]).T  # shape: [n_samples, n_classifiers]

            # for each sample, count votes (with weights if given)
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1, arr=predictions
            )

        # convert back to original labels
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """Predict class probabilities (average across classifiers)."""
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0,
                               weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """Get classifier parameter names for GridSearchCV."""
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out
