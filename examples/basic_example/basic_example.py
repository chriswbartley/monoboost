"""
======================================
How to fit a basic model
======================================

These examples show how to fit a model using MonoBoost. There are two model types: `MonoBoost`, and `MonoBoostEnsemble`. `MonoBoost` sequentially fits `num_estimators` partially monotone cone rules to the dataset using gradient boosting. `MonoBoostEnsemble` fits a sequence of  `MonoBoost` classifiers each of size `learner_num_estimators` (up to a total of `num_estimators`) using gradient boosting. The advantage of `MonoBoostEnsemble` is to allow the added feature of stochastic subsampling of fraction `sample_fract` after every `learner_num_estimators` cones.
"""

import numpy as np
import monoboost as mb
from sklearn.datasets import load_boston

###############################################################################
# Load the data
# ----------------
#
# First we load the standard data source on `Boston Housing <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`_, and convert the output from real valued (regression) to binary classification with roughly 50-50 class distribution:
#

data = load_boston()
y = data['target']
X = data['data']
features = data['feature_names']

###############################################################################
# Specify the monotone features
# -------------------------
#
# There are 13 predictors for house price in the Boston dataset:
# 1. CRIM - per capita crime rate by town
# 2. ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS - proportion of non-retail business acres per town.
# 4. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 5. NOX - nitric oxides concentration (parts per 10 million)
# 6. RM - average number of rooms per dwelling
# 7. AGE - proportion of owner-occupied units built prior to 1940
# 8. DIS - weighted distances to five Boston employment centres
# 9. RAD - index of accessibility to radial highways
# 10. TAX - full-value property-tax rate per $10,000
# 11. PTRATIO - pupil-teacher ratio by town
# 12. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13. LSTAT - % lower status of the population
#
###############################################################################
# The output is MEDV - Median value of owner-occupied homes in $1000's, but we convert it to a binary y in +/-1 indicating whether MEDV is less than $21(,000):

y[y< 21]=-1 # convert real output to 50-50 binary classification
y[y>=21]=+1 

###############################################################################
# We suspect that the number of rooms (6. RM) and the highway 
# accessibility (9. RAD) would, if anything, increase the price of a house
# (all other things being equal). Likewise we suspect that crime rate (1.
# CRIM), distance from employment (8. DIS) and percentage of lower status residents (13. LSTAT) would be likely to, if anything, decrease house prices. So we have:

incr_feats=[6,9]
decr_feats=[1,8,13]

###############################################################################
# Specify and fit a MonoBoost model
# -------------------------
# We now initialise our classifier:

# Specify hyperparams for model solution
vs = [0.01, 0.1, 0.2, 0.5, 1]
eta = 0.25
learner_type = 'two-sided'
num_estimators = 10
# Solve model
mb_clf = mb.MonoBoost(n_feats=X.shape[1], incr_feats=incr_feats,
                          decr_feats=decr_feats, num_estimators=num_estimators,
                          fit_algo='L2-one-class', eta=eta, vs=vs,
                          verbose=False, learner_type=learner_type)
mb_clf.fit(X, y)
# Assess the model
y_pred = mb_clf.predict(X)
acc = np.sum(y == y_pred) / len(y)

###############################################################################
# Specify and fit a MonoBoostEnsemble model
# -------------------------
# We now initialise our classifier:

# Specify hyperparams for model solution
vs = [0.01, 0.1, 0.2, 0.5, 1]
eta = 0.25
learner_type = 'one-sided'
num_estimators = 10
learner_num_estimators = 2
learner_eta = 0.25
learner_v_mode = 'random'
sample_fract = 0.5
random_state = 1
standardise = True
# Solve model
mb_clf = mb.MonoBoostEnsemble(
    n_feats=X.shape[1],
    incr_feats=incr_feats,
    decr_feats=decr_feats,
    num_estimators=num_estimators,
    fit_algo='L2-one-class',
    eta=eta,
    vs=vs,
    verbose=False,
    learner_type=learner_type,
    learner_num_estimators=learner_num_estimators,
    learner_eta=learner_eta,
    learner_v_mode=learner_v_mode,
    sample_fract=sample_fract,
    random_state=random_state,
    standardise=standardise)
mb_clf.fit(X, y)
# Assess the model (MonoBoostEnsemble)
y_pred = mb_clf.predict(X)
acc = np.sum(y == y_pred) / len(y)

###############################################################################
# Final notes
# -----------------------
# In a real scenario we would use a hold out technique such as cross-validation 
# to tune the hyperparameters `v`, `eta` and `num_estimators` but this is 
# standard practice and not covered in these basic examples. Note that for 
# tuning `num_estimators` we can use `predict(X,cum=True)` because as standard 
# for boosting, the stagewise predictions are stored. Enjoy!
