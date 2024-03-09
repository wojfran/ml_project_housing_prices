from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# In this example the transformer has one hyperparameter, add_bedrooms_per_room,
# set to True by default (it is often helpful to provide sensible defaults). This hyperpara‐
# meter will allow you to easily find out whether adding this attribute helps the
# Machine Learning algorithms or not. More generally, you can add a hyperparameter
# to gate any data preparation step that you are not 100% sure about. The more you
# automate these data preparation steps, the more combinations you can automatically
# try out, making it much more likely that you will find a great combination (and sav‐
# ing you a lot of time).

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        

