###############################################################################
# How to fit a basic model
# ----------------
#
# These examples show how to fit a model using MonoBoost. There are two model types: `MonoBoost`, and `MonoBoostEnsemble`. `MonoBoost` sequentially fits `num_estimators` partially monotone cone rules to the dataset using gradient boosting. `MonoBoostEnsemble` fits a sequence of  `MonoBoost` classifiers each of size `learner_num_estimators` (up to a total of `num_estimators`) using gradient boosting. The advantage of `MonoBoostEnsemble` is to allow the added feature of stochastic subsampling of fraction `sample_fract` after every `learner_num_estimators` cones.
#

import numpy as np
import monoboost as mb
from sklearn.datasets import load_boston

###############################################################################
# Load the data
# ----------------
#
# First we load the standard data source on `Boston Housing <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`_.
#

    data = load_boston()
    y = data['target']
    X = data['data']
    features = data['feature_names']

###############################################################################
# Specify the monotone features
# ----------------
#
# There are 14 predictors for house price in the Boston dataset:
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
MEDV - Median value of owner-occupied homes in $1000's
#
# We suspect that the number of rooms (6. RM) and the highway 
# accessibility (9. RAD) would, if anything, increase the price of a house
# (all other things being equal). Likewise we suspect that crime rate (1.
# CRIM), distance from employment (8. DIS) and percentage of lower status residents (13. LSTAT) would be likely to, if anything, decrease house prices. So we have `incr_feats=[6,9]` and `decr_feats=[1,8,13]`.

    data = load_boston()
    y = data['target']
    X = data['data']
    features = data['feature_names']

    multi_class = False
    # Specify monotone features
    incr_feat_names = ['RM', 'RAD']
    decr_feat_names = ['CRIM', 'DIS', 'LSTAT']
    # get 1 based indices of incr and decr feats
    incr_feats = [i + 1 for i in np.arange(len(features)) if
                  features[i] in incr_feat_names]
    decr_feats = [i + 1 for i in np.arange(len(features)) if
                  features[i] in decr_feat_names]
    # Convert to classification problem
    if multi_class:
        y_class = y.copy()
        thresh1 = 15
        thresh2 = 21
        thresh3 = 27
        y_class[y > thresh3] = 3
        y_class[np.logical_and(y > thresh2, y <= thresh3)] = 2
        y_class[np.logical_and(y > thresh1, y <= thresh2)] = 1
        y_class[y <= thresh1] = 0
    else:  # binary
        y_class = y.copy()
        thresh = 21  # middle=21
        y_class[y_class < thresh] = -1
        y_class[y_class >= thresh] = +1
    return X[0:max_N, :], y_class[0:max_N], incr_feats, decr_feats


# Load data
X, y, incr_feats, decr_feats = load_data_set()

###############################################################################


# File headers and naming
# -----------------------
# Sphinx-gallery files must be initialized with a header like the one above.
# It must exist as a part of the triple-quotes docstring at the start of the
# file, and tells SG the title of the page. If you wish, you can include text
# that comes after the header, which will be rendered as a contextual bit of
# information.
#
# In addition, if you want to render a file with sphinx-gallery, it must match
# the file naming structure that the gallery is configured to look for. By
# default, this is `plot_*.py`.
#
# Interweaving code with text
# ---------------------------
#
# Sphinx-gallery allows you to interweave code with your text. For example, if
# put a few lines of text below...

N = 1000

# They will be rendered as regular code. Note that now I am typing in a
# comment, because we've broken the chain of commented lines above.
x = np.random.randn(N)

# If we want to create another formatted block of text, we need to add a line
# of `#` spanning the whole line below. Like this:

###############################################################################
# Now we can once again have nicely formatted $t_{e}\chi^t$!

# Let's create our y-variable so we can make some plots
y = .2 * x + .4 * np.random.randn(N)

###############################################################################
# Plotting images
# ---------------
#
# Sphinx-gallery captures the images generated by matplotlib. This means that
# we can plot things as normal, and these images will be grouped with the
# text block that the fall underneath. For example, we could plot these two
# variables and the image will be shown below:

fig, ax = plt.subplots()
ax.plot(x, y, 'o')

###############################################################################
# Multiple images
# ---------------
#
# If we want multiple images, this is easy too. Sphinx-gallery will group
# everything together that's within the latest text block.

fig, axs = plt.subplots(1, 2)
axs[0].hist(x, bins=20)
axs[1].hist(y, bins=20)

fig, ax = plt.subplots()
ax.hist2d(x, y, bins=20)

###############################################################################
# Other kinds of formatting
# -------------------------
#
# Remember, rST can do all kinds of other cool stuff. We can even do things
# like add references to other packages and insert images. Check out this
# `guide <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_ for
# some sample rST code.
#
# .. image:: http://www.sphinx-doc.org/en/stable/_static/sphinxheader.png
#   :width: 80%
#
# In the meantime, enjoy sphinx-gallery!
