import fetch_data
import manipulate_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import custom_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV
from scipy import stats

fetch_data.fetch_housing_data()
housing = manipulate_data.load_housing_data()

head = housing.head()

print(head)

# housing.hist(bins=50, figsize=(20,15))
# plt.show()

housing["income_cat"] = pd.cut(housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5])

# Sampling the test set in a stratified way, to generate a test set indicative
# of the average distribution of income cathegories. Fe if 20% of all districts lie
# in cathegory 1 then about 20% of districts in the test set should also be in
# that cathegory
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
plt.show()

# Checking for some additional features that could be benefitial for the training
# of the algorithm. It seems that the bedrooms per room and rooms per household 
# features would be the most indicative of the median house price right after
# the median income.
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Filling the missing feature vaues by their respective median computed by sklearn
# imputter. This is one of the three options, the other two would be to either drop
# the rows with the null values "housing.dropna(subset=["total_bedrooms"]) " or
# drop the feature as a whole: "housing.drop("total_bedrooms", axis=1) "
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)

# Transforming the data back into a Pandas DataFrame from a plain numPy array
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# INVESTIGATING THE ONLY TEXT ATTRIBUTE "ocean_proximity"
# Since most ML algorithms prefer to work on numerical values it might be worth 
# to convert those to numbers, this is easy since they are already cathegorized and aren't
# arbitrary

housing_cat = housing[["ocean_proximity"]]
print (housing_cat.head(10))

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(ordinal_encoder.categories_)

# Such an encoding  0 == '<1H OCEAN', 1 == 'INLAND', 2 == 'ISLAND', 3 == 'NEAR BAY', 4 == 'NEAR OCEAN'
# May be misleading for an algorithm as it will assume that two nearby values are more 
# similar than two distant values. This may be fine in some cases (e.g., for ordered 
# categories such as “bad,” “average,” “good,” and “excellent”), but it is obviously 
# not the case for the ocean_proximity column (for example, categories 0 and 4 are 
# clearly more similar than categories 0 and 1).

# To fix this issue, a common solution is to create one binary attribute per category: 
# one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise), another 
# attribute equal to 1 when the category is “INLAND” (and 0 otherwise), and so on. This 
# is called one-hot encoding, because only one attribute will be equal to 1 (hot), while 
# the others will be 0 (cold). The new attributes are sometimes called dummy attributes. 
# Scikit-Learn provides a OneHotEncoder class to convert categorical values into one-hot 
# vectors.

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

housing_cat_1hot = housing_cat_1hot.toarray()

# Although Scikit-Learn provides many useful transformers, you will need to write
# your own for tasks such as custom cleanup operations or combining specific
# attributes. You will want your transformer to work seamlessly with Scikit-Learn func‐
# tionalities (such as pipelines), and since Scikit-Learn relies on duck typing (not inher‐
# itance), all you need to do is create a class and implement three methods: fit()
# (returning self), transform(), and fit_transform().

# You can get the last one for free by simply adding TransformerMixin as a base class.
# If you add BaseEstimator as a base class (and avoid *args and **kargs in your con‐
# structor), you will also get two extra methods (get_params() and set_params()) that
# will be useful for automatic hyperparameter tuning.
# Example in custom_transformer.py

attr_adder = custom_transformer.CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# One of the most important transformations you need to apply to your data is feature
# scaling. With few exceptions, Machine Learning algorithms don’t perform well when
# the input numerical attributes have very different scales. This is the case for the hous‐
# ing data: the total number of rooms ranges from about 6 to 39,320 (I think its rooms per 
# district), while the median incomes only range from 0 to 15. Note that scaling the target 
# values is generally not required.

# There are two common ways to get all attributes to have the same scale: min-max
# scaling and standardization.

# Min-max scaling (many people call this normalization) is the simplest: values are shif‐
# ted and rescaled so that they end up ranging from 0 to 1. We do this by subtracting
# the min value and dividing by the max minus the min. Scikit-Learn provides a trans‐
# former called MinMaxScaler for this. It has a feature_range hyperparameter that lets
# you change the range if, for some reason, you don’t want 0–1.

# Standardization is different: first it subtracts the mean value (so standardized values
# always have a zero mean), and then it divides by the standard deviation so that the
# resulting distribution has unit variance. Unlike min-max scaling, standardization
# does not bound values to a specific range, which may be a problem for some algo‐
# rithms (e.g., neural networks often expect an input value ranging from 0 to 1). How‐
# ever, standardization is much less affected by outliers. For example, suppose a district
# had a median income equal to 100 (by mistake). Min-max scaling would then crush
# all the other values from 0–15 down to 0–0.15, whereas standardization would not be
# much affected. Scikit-Learn provides a transformer called StandardScaler for
# standardization.


# Explanation on page 100 and 101 in the pdf but basically a pipeline condesnes
# all of the transformers we have been using into one bigger transformer, which
# is convenient

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', custom_transformer.CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
 ])

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
 scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
               
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, 
                             scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

joblib.dump(lin_reg, "lin_reg.pkl")
joblib.dump(tree_reg, "tree_reg.pkl")
joblib.dump(forest_reg, "forest_reg.pkl")

# To load models:
# my_model_loaded = joblib.load("my_model.pkl")

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# The grid search approach is fine when you are exploring relatively few combinations,
# like in the previous example, but when the hyperparameter search space is large, it is
# often preferable to use RandomizedSearchCV instead. This class can be used in much
# the same way as the GridSearchCV class, but instead of trying out all possible combi‐
# nations, it evaluates a given number of random combinations by selecting a random
# value for each hyperparameter at every iteration. This approach has two main
# benefits:
# • If you let the randomized search run for, say, 1,000 iterations, this approach will
# explore 1,000 different values for each hyperparameter (instead of just a few val‐
# ues per hyperparameter with the grid search approach).
# • Simply by setting the number of iterations, you have more control over the com‐
# puting budget you want to allocate to hyperparameter search.

# Testing the on the test set:
    
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2

# In some cases, such a point estimate of the generalization error will not be quite
# enough to convince you to launch: what if it is just 0.1% better than the model cur‐
# rently in production? You might want to have an idea of how precise this estimate is.
# For this, you can compute a 95% confidence interval for the generalization error using
# scipy.stats.t.interval():

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, 
                                               loc=squared_errors.mean(), 
                                               scale=stats.sem(squared_errors)))

print(confidence_interval)