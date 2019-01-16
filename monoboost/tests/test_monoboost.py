from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
import monoboost as mb
from sklearn.datasets import load_boston
import time
from sklearn.metrics import  confusion_matrix
def load_data_set():
    # Load data

    data = load_boston()
    y = data['target']
    X = data['data']
    features = data['feature_names']
    # Specify monotone features
    incr_feat_names = ['RM']#['RM', 'RAD']
    decr_feat_names = ['CRIM', 'LSTAT'] # ['CRIM', 'DIS', 'LSTAT']
    # get 1 based indices of incr and decr feats
    incr_feats = [i + 1 for i in np.arange(len(features)) if
                  features[i] in incr_feat_names]
    decr_feats = [i + 1 for i in np.arange(len(features)) if
                  features[i] in decr_feat_names]
    # Convert to classification problem
    # Multi-class
    y_multiclass = y.copy()
    thresh1 = 15
    thresh2 = 21
    thresh3 = 27
    y_multiclass[y > thresh3] = 3
    y_multiclass[np.logical_and(y > thresh2, y <= thresh3)] = 2
    y_multiclass[np.logical_and(y > thresh1, y <= thresh2)] = 1
    y_multiclass[y <= thresh1] = 0
    y_multiclass =np.asarray(y_multiclass ,dtype=np.int32)
    # Binary
    y_binary = y.copy()
    thresh = 21  # middle=21
    y_binary[y_binary < thresh] = -1
    y_binary[y_binary >= thresh] = +1
    y_binary =np.asarray(y_binary ,dtype=np.int32)
    return X, y_binary, y_multiclass, incr_feats, decr_feats


# Load data
max_N = 200#200
np.random.seed(13) # comment out for changing random training set
X, y_binary, y_multiclass, incr_feats, decr_feats = load_data_set()
indx_train=np.random.permutation(np.arange(X.shape[0]))[0:max_N]
inx_test=np.asarray([i for i in np.arange(max_N) if i not in indx_train ])
X_train=X[indx_train,:]
X_test=X[inx_test,:]


y_train=dict()
y_test=dict()
n_classes=dict()
y_train['binary']=y_binary[indx_train]
y_train['multiclass']=y_multiclass[indx_train]
y_test['binary']=y_binary[inx_test]
y_test['multiclass']=y_multiclass[inx_test]
n_classes['binary']=2
n_classes['multiclass']=4


def test_model_fit():
    # Specify hyperparams for model solution
    vs = [0.01, 0.1, 0.2, 0.5, 1]
    etas = [0.06, 0.125, 0.25, 0.5, 1]
    eta =  etas[2]
    learner_type = 'two-sided'
    max_iters = 100
    fit_algo='none'#'L2-one-class' none
    # store benchmark
    acc_benchmark={'multiclass': 0.64000000000000001, 'binary': 0.784}
    for response in ['binary']:#'multiclass','binary']:#,'multiclass']:
        y_train_=y_train[response]
        y_test_=y_test[response]
        # Solve model
        mb_clf = mb.MonoBoost(n_feats=X.shape[1], incr_feats=incr_feats,
                              decr_feats=decr_feats, num_estimators=max_iters,
                              fit_algo=fit_algo, eta=eta, vs=vs,
                              verbose=False, learner_type=learner_type)
        mb_clf.fit(X_train, y_train_)
        # Assess fit
        y_pred = mb_clf.predict(X_test)
        cm=confusion_matrix(y_test_,y_pred)
        acc = np.sum(y_test_ == y_pred) / len(y_test_)
        print(acc)
        #npt.assert_almost_equal(acc, 0.70999999999)
        return mb_clf

#def test_ensemble_model_fit():
#    # Specify hyperparams for model solution
#    vs = [0.01, 0.1, 0.2, 0.5, 1]
#    etas = [0.06, 0.125, 0.25, 0.5, 1]
#    eta = 0.5#etas[2]
#    learner_type = 'two-sided'
#    num_estimators = 20
#    learner_num_estimators = 5
#    learner_eta = 1#0.5#1.0 #0.5 # needs to be higher to get useful cones on each monoboost iteration, otherwise (for eta<1) keeps repeating the same cones to reduce the RMSE base monoboost.
#    learner_v_mode = 'all'#'random'
#    sample_fract = 0.6
#    random_state = 1
#    standardise = False
#    for response in ['multiclass']:#,'multiclass']: binary
#        y_train_=y_train[response]
#        y_test_=y_test[response]
#        # Solve model
#        mb_clf = mb.MonoBoostEnsemble(
#            n_feats=X.shape[1],
#            incr_feats=incr_feats,
#            decr_feats=decr_feats,
#            num_estimators=num_estimators,
#            fit_algo='L2-one-class',
#            eta=eta,
#            vs=vs,
#            verbose=False,
#            learner_type=learner_type,
#            learner_num_estimators=learner_num_estimators,
#            learner_eta=learner_eta,
#            learner_v_mode=learner_v_mode,
#            sample_fract=sample_fract,
#            random_state=random_state,
#            standardise=standardise)
#        mb_clf.fit(X_train, y_train_)
#        # Assess fit
#        y_pred = mb_clf.predict(X_test)
#        cm=confusion_matrix(y_test_,y_pred)
#        acc = np.sum(y_test_ == y_pred) / len(y_test_)
#        print(acc)
#        return mb_clf
#        #npt.assert_almost_equal(acc, 0.6959999999)

#mbe_clf=test_ensemble_model_fit()
mb_clf=test_model_fit()