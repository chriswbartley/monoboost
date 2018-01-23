## monoboost
[![Build Status](https://travis-ci.org/chriswbartley/monoboost.svg?branch=master)](https://travis-ci.org/chriswbartley/monoboost)
[![Appveyor Status](https://ci.appveyor.com/api/projects/status/github/chriswbartley/monoboost)](https://ci.appveyor.com/project/chriswbartley/monoboost)
[![RTD Status](https://readthedocs.org/projects/monoboost/badge/?version=latest
)](https://readthedocs.org/projects/monoboost/badge/?version=latest)

Monoboost is an instance based classifier with *partial* monotonicity capability (i.e. the ability to specify non-monotone features). It uses standard inequality constraints for the monotone features, and novel L1 cones to cater for the non-monotone features. This package contains two classifiers: `MonoBoost()`, and `MonoBoostEnsemble()`. 

You are more than welcome to make use of this code for research purposes. If so, please cite:
Bartley C., Liu W., Reynolds M., 2018, A Novel Framework for Partially Monotone Rule Ensembles. ICDE 2018 prepub Paris, France, April 16-20, 2018, IEEE Computer Society. It is available in PDF [here](http://staffhome.ecm.uwa.edu.au/~19514733/). 


### Code Example
First we define the monotone features, using the corresponding one-based `X` array column indices:
```
incr_feats=[6,9]
decr_feats=[1,8,13]
```
The specify the hyperparameters (see original paper for explanation):
```
vs = [0.01, 0.1, 0.2, 0.5, 1]
eta = 0.25
learner_type = 'two-sided'
num_estimators = 10
```
And initialise and solve the classifier using `scikit-learn` norms:
```
mb_clf = mb.MonoBoost(n_feats=X.shape[1], incr_feats=incr_feats,
                          decr_feats=decr_feats, num_estimators=num_estimators,
                          fit_algo='L2-one-class', eta=eta, vs=vs,
                          verbose=False, learner_type=learner_type)
mb_clf.fit(X, y)
y_pred = mb_clf.predict(X)
```	
Of course usually the above will be embedded in some estimate of generalisation error such as cross-validation. Note however that since it is based on gradient boosting, `num_estimators` can be estimated using the cumulative predictions after each estimator is added, which can be accessed efficiently using `mb_clf.predict(X,cum=True)`.

For more examples including for MonoBoostEnsemble, see [the documentation](http://monoboost.readthedocs.io/en/latest/index.html).

### Installation

To install, simply use:
```
pip install monoboost
```

### Documentation

Documentation is provided [here](http://monoboost.readthedocs.io/en/latest/index.html).

### Contributors

Pull requests welcome! This is a proof of concept implementation and in need of optimisation. Notes:
 - We use the
[PEP8 code formatting standard](https://www.python.org/dev/peps/pep-0008/), and
we enforce this by running a code-linter called
[`flake8`](http://flake8.pycqa.org/en/latest/) during continuous integration.
 - Continuous integration is used to run the tests in `/monoboost/tests/test_monoboost.py`, using [Travis](https://travis-ci.org/chriswbartley/monoboost.svg?branch=master) (Linux) and [Appveyor](https://ci.appveyor.com/api/projects/status/github/chriswbartley/monoboost) (Windows).
 
### License
BSD 3 Clause, Copyright (c) 2017, Christopher Bartley
All rights reserved.
