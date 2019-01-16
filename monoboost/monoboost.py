# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:54:18 2017

@author: 19514733
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from cvxopt import matrix as cvxmat, solvers
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
import monoboost


#__all__ = [
#    "Scale",
#    "MonoComparator",
#    "MonoLearner",
#    "MonoBoost",
#    "MonoBoostEnsemble",
#    "apply_rules_c"]

TOL = 0  # 1e-55


class Scale():
    """Performs scaling of linear variables according to Friedman et al. 2005
    Sec 5

    Each variable is firsst Winsorized l->l*, then standardised as 0.4 x l* /
    std(l*).
    Warning: this class should not be used directly.
    """

    def __init__(self, trim_quantile=0.0):
        self.trim_quantile = trim_quantile
        self.scale_multipliers = None
        self.winsor_lims = None

    def train(self, X):
        # get winsor limits
        self.winsor_lims = np.ones([2, X.shape[1]]) * np.inf
        self.winsor_lims[0, :] = -np.inf
        if self.trim_quantile > 0:
            for i_col in np.arange(X.shape[1]):
                lower = np.percentile(X[:, i_col], self.trim_quantile * 100)
                upper = np.percentile(
                    X[:, i_col], 100 - self.trim_quantile * 100)
                self.winsor_lims[:, i_col] = [lower, upper]
        # get multipliers
        scale_multipliers = np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals = len(np.unique(X[:, i_col]))
            # don't scale binary variables which are effectively already rules:
            if num_uniq_vals > 2:
                X_col_winsorised = X[:, i_col].copy()
                X_col_winsorised[X_col_winsorised <
                                 self.winsor_lims[0, i_col]
                                 ] = self.winsor_lims[0, i_col]
                X_col_winsorised[X_col_winsorised >
                                 self.winsor_lims[1, i_col]
                                 ] = self.winsor_lims[1, i_col]
                scale_multipliers[i_col] = 1.0 / np.std(X_col_winsorised)
        self.scale_multipliers = scale_multipliers

    def scale(self, X):
        return X * self.scale_multipliers

    def unscale(self, X):
        return X / self.scale_multipliers


class MonoComparator():
    def __init__(self, n_feats, incr_feats, decr_feats, nmt_hyperplane=None):
        self.incr_feats = np.asarray(incr_feats)
        self.decr_feats = np.asarray(decr_feats)
        self.nmt_hyperplane = nmt_hyperplane
        self.mt_feats = np.asarray(list(incr_feats) + list(decr_feats))
        self.nmt_feats = np.asarray(
            [j for j in np.arange(n_feats) + 1 if j not in self.mt_feats])
        self.n_feats = n_feats

    def compare(self, x1_in, x2_in, check_nmt_feats=True, strict=False):
        # returns: -1 if decreasing, 0 if identical, +1 if increasing, -99 if
        # incomparable
        if len(self.mt_feats) == 0:
            return -99
        elif len(x1_in.shape) > 1:
            x1 = np.ravel(x1_in)
            x2 = np.ravel(x2_in)
        else:
            x1 = x1_in.copy()
            x2 = x2_in.copy()
        # check for identical
        if np.array_equal(x1, x2):
            return 0
        # reverse polarity of decreasing features
        for dec_feat in self.decr_feats:
            x1[dec_feat - 1] = -1 * x1[dec_feat - 1]
            x2[dec_feat - 1] = -1 * x2[dec_feat - 1]
        # check mt feats all increasing (or decreasing)
        mt_feats_difference = np.zeros(self.n_feats)
        if len(self.mt_feats) > 0:
            feats_indx = self.mt_feats - 1
            mt_feats_difference[feats_indx] = x2[feats_indx] - x1[feats_indx]
        mt_feats_same = np.sum(mt_feats_difference[self.mt_feats - 1] == 0)
        if strict:
            mt_feats_incr = np.sum(mt_feats_difference[self.mt_feats - 1] > 0)
            mt_feats_decr = np.sum(mt_feats_difference[self.mt_feats - 1] < 0)
        else:
            mt_feats_incr = np.sum(mt_feats_difference[self.mt_feats - 1] >= 0)
            mt_feats_decr = np.sum(mt_feats_difference[self.mt_feats - 1] <= 0)
        if mt_feats_same == len(self.mt_feats):
            comp = 0
        elif mt_feats_incr == len(self.mt_feats):  # increasing
            comp = +1
        elif mt_feats_decr == len(self.mt_feats):  # decreasing
            comp = -1
        else:  # incomparale
            comp = -99
        # short exit if available
        if comp == -99 or comp == 0:
            return -99
        # if still going, check mt feats by weakened planes
        if len(
            self.nmt_feats) == 0 or not check_nmt_feats or (
                self.nmt_hyperplane is None):
            nmt_feat_compliance = True
        else:
            x_diff = np.abs(x2 - x1)
            dot_prod = np.dot(self.nmt_hyperplane, x_diff)
            nmt_feat_compliance = dot_prod >= -TOL
        # return result
        if nmt_feat_compliance:
            return comp
        else:  # incomparable due to nmt features
            return -99

def calc_loss_deviance(curr_ttls,y,lidstone=0.01):
    
#    p_ = np.exp(curr_ttls) / \
#        (np.exp(curr_ttls) + np.exp(-curr_ttls))
#    #lidstone=0.01
#    p_=(p_*len(y)+lidstone)/(len(y)+2*lidstone) # protect against perfect probabilities causing instability
#    loss_=-np.sum(y*np.log(p_))-np.sum((1-y)*np.log(1-p_) )
    
    loss_=monoboost.calc_loss_deviance_c(curr_ttls,y,lidstone)
    return loss_

class MonoLearner():
    def __init__(
            self,
            n_feats,
            incr_feats,
            decr_feats,
            coefs=None,
            dirn=None,
            x_base=None,
            nmt_hyperplane=None,
            learner_type='two-sided',
            loss='rmse'):
        self.incr_feats = np.asarray(incr_feats)
        self.decr_feats = np.asarray(decr_feats)
        self.coefs = coefs
        self.dirn = dirn
        self.x_base = x_base
        self.intercept_=0.
        self.mt_feats = np.asarray(list(incr_feats) + list(decr_feats))
        self.nmt_feats = np.asarray(
            [j for j in np.arange(n_feats) + 1 if j not in self.mt_feats])
        self.mt_feat_types=np.zeros(n_feats,dtype=np.float64)
        if len(self.incr_feats)>0:
            self.mt_feat_types[self.incr_feats-1]=+1.
        if len(self.decr_feats)>0:
            self.mt_feat_types[self.decr_feats-1]=-1.
        self.comparator = MonoComparator(
            n_feats, incr_feats, decr_feats, nmt_hyperplane)
        self.nmt_hyperplane = nmt_hyperplane
        self.learner_type_code = 0 if learner_type == 'two-sided' else 1
        # note loss only affects the calculation of the coefficients - all
        # splits are done RMSE
        self.loss = loss
        

    @property
    def nmt_hyperplane(self):
        """I'm the 'x' property."""

        return self.comparator.nmt_hyperplane

    @nmt_hyperplane.setter
    def nmt_hyperplane(self, value):
        self.comparator.nmt_hyperplane = value
        
    def get_comparable_points(self,X):

        intercepts=np.asarray([self.intercept_],dtype=np.float64)

        if len(X.shape)<2:
            X_=np.asarray(X.reshape([1,-1]),dtype=np.float64)
        else:
            X_=np.asarray(X,dtype=np.float64)

        X_base_pts_=np.asarray(self.x_base.reshape([1,-1]),dtype=np.float64)


        nmt_hps_=np.asarray(self.nmt_hyperplane.reshape([1,-1]),dtype=np.float64)
   
        X_comp_pts=np.zeros([X_.shape[0],X_base_pts_.shape[0]],dtype=np.int32)
        monoboost.apply_rules_c(X_,
                  X_base_pts_, 
                  nmt_hps_,
                  intercepts,
                  self.mt_feat_types,
                  np.float64(self.dirn),
                  0,
                  X_comp_pts)
        return X_comp_pts[:,0]==1    
    def decision_function(self, X_pred):
        if len(X_pred.shape)==1:
            X_pred_=np.asarray(X_pred.reshape([1,-1]),dtype=np.float64)
        else:
            X_pred_=np.asarray(X_pred,dtype=np.float64)
            
        dirn=self.dirn
        X_rule_transform_=np.zeros([X_pred.shape[0],1],dtype=np.int32)
    
        monoboost.apply_rules_c(X_pred_,
              np.asarray(self.x_base.reshape([1,-1]),dtype=np.float64), 
              np.asarray(self.nmt_hyperplane.reshape([1,-1]),dtype=np.float64),
              np.asarray([self.intercept_],dtype=np.float64),
              self.mt_feat_types,
              np.float64(dirn),
              0,
               X_rule_transform_)
        X_rule_transform_=X_rule_transform_.ravel()
        is_comp=X_rule_transform_
        y_pred = np.zeros(X_pred_.shape[0])
        y_pred[X_rule_transform_==1]=self.coefs[1]
        y_pred[X_rule_transform_==0]=self.coefs[0]
        return [y_pred,is_comp]
    
#    def predict_proba(self, X_pred):
#        if len(X_pred.shape) == 1:
#            X_pred_ = np.zeros([1, len(X_pred)])
#            X_pred_[0, :] = X_pred
#        else:
#            X_pred_ = X_pred
#
#        y_pred = np.zeros(X_pred_.shape[0])
#        is_comp = np.zeros(X_pred_.shape[0])
#        for i in np.arange(len(y_pred)):
#            comp = self.comparator.compare(self.x_base, X_pred_[i, :])
#            is_comp[i] = 1 if comp == 0 or comp == self.dirn else 0
#            y_pred[i] = self.coefs[1] if (
#                comp == 0 or comp == self.dirn) else self.coefs[0]
#        return [y_pred, is_comp]


    def calc_coefs(self,comp_pts,hp,res_train,y,curr_ttls):
#        incomp_pts = np.asarray(np.setdiff1d(
#        np.arange(len(y)), comp_pts))
#        res_comp_pts = res_train[comp_pts]
#        res_incomp_pts = res_train[incomp_pts]
        
        if self.loss == 'deviance':
#            incomp_pts = np.asarray(np.setdiff1d(
#            np.arange(len(y)), comp_pts))
#            res_comp_pts = res_train[comp_pts]
#            res_incomp_pts = res_train[incomp_pts]
#            
#            sum_res_comp_ = np.sum(np.abs(res_comp_pts) * (
#                1 - np.abs(res_comp_pts)))
#            sum_res_incomp_ = np.sum(np.abs(res_incomp_pts) * (
#                1 - np.abs(res_incomp_pts)))
#            signed_sum_res_comp_ = np.sum(res_comp_pts)
#            signed_sum_res_incomp_ = np.sum(res_incomp_pts)
            
            [[sum_res_comp, signed_sum_res_comp,sum_res_incomp,signed_sum_res_incomp],comp_indx]=monoboost.get_signed_sums_c(comp_pts,res_train)
            
            if (sum_res_comp > 1e-9 and
                    sum_res_incomp > 1e-9 and
                    np.abs(signed_sum_res_comp) > 1e-9 and
                    np.abs(signed_sum_res_incomp) > 1e-9):
                coef_in = 0.5 * signed_sum_res_comp / \
                    (sum_res_comp)
                coef_out = 0.5 * signed_sum_res_incomp / \
                    (sum_res_incomp)                
                ratio =np.abs(coef_in / coef_out) # monoboost.get_max_ratio0.5# np.max(
                if ratio<1:
                    ratio=1./ratio
                    #[np.abs(coef_in / coef_out),
                    # np.abs(coef_out / coef_in)])
                
            else:
                coef_in = 0
                coef_out = 0
                ratio = 0.


        elif self.loss == 'rmse':
            raise NotImplemented()
#            #use_M-regression (huber loss)
#            use_huber=True
#            if use_huber:
#                q_alpha=0.5
#                q_in=np.percentile(np.abs(y[comp_pts] - curr_totals[comp_pts]),q_alpha)
#                res_in=y[comp_pts] - curr_totals[comp_pts]
#                median_in=np.median(res_in)
#                coef_in = median_in + (
#                            1/len(comp_pts)*(np.sum(np.sign(res_in-
#                                 median_in)*np.min(np.hstack([q_in*np.ones(len(res_in)).reshape([-1,1]),np.abs(res_in-
#                                 median_in).reshape([-1,1])]),axis=1))))
#
#   
#                if self.learner_type_code == 1:
#                    coef_out=0
#                else:
#                    q_out=np.percentile(np.abs(y[incomp_pts] - curr_totals[incomp_pts]),q_alpha)
#                    res_out=y[incomp_pts] - curr_totals[incomp_pts]
#                    median_out=np.median(res_out) 
#                    coef_out = median_out + (
#                            1/len(incomp_pts)*(np.sum(np.sign(res_out-
#                                 median_out)*np.min(np.hstack([q_out*np.ones(len(res_out)).reshape([-1,1]),np.abs(res_out-
#                                 median_out).reshape([-1,1])]),axis=1))))
#            else:
#                coef_in = np.median(
#                    y[comp_pts] - curr_totals[comp_pts])                                
#                coef_out = (0 if self.learner_type_code == 1 else
#                        np.median(y[incomp_pts] -
#                                  curr_totals[incomp_pts])) 


        # calc loss
#        new_preds=curr_ttls.copy()
#        new_preds[comp_pts]+=coef_in
#        new_preds[comp_indx==0]+=coef_out
#        
        #new_preds=curr_ttls.copy()
        new_preds=np.zeros(len(curr_ttls),dtype=np.float64)
        monoboost.update_preds_2_c(new_preds,curr_ttls,comp_indx,coef_in,coef_out)
        
        
        loss_=self.calc_loss(new_preds,y)
        
        # one-sided check
        if self.learner_type_code == 1: 
            coef_out=0.
            
        return [coef_out,coef_in,ratio,loss_]
    
    def calc_coefs_old(self,comp_pts,hp,res_train,y,curr_ttls):
        incomp_pts = np.asarray(np.setdiff1d(
        np.arange(len(y)), comp_pts))
        res_comp_pts = res_train[comp_pts]
        res_incomp_pts = res_train[incomp_pts]
        mean_res_in = np.mean(res_comp_pts)
        mean_res_out = np.mean(res_incomp_pts)

        if self.loss == 'deviance':
            
            sum_res_comp = np.sum(np.abs(res_comp_pts) * (
                1 - np.abs(res_comp_pts)))
            sum_res_incomp = np.sum(np.abs(res_incomp_pts) * (
                1 - np.abs(res_incomp_pts)))
            signed_sum_res_comp = np.sum(res_comp_pts)
            signed_sum_res_incomp = np.sum(res_incomp_pts)
            if (sum_res_comp > 1e-9 and
                    sum_res_incomp > 1e-9 and
                    np.abs(signed_sum_res_comp) > 1e-9 and
                    np.abs(signed_sum_res_incomp) > 1e-9):
                coef_in = 0.5 * signed_sum_res_comp / \
                    (sum_res_comp)
                coef_out = 0.5 * signed_sum_res_incomp / \
                    (sum_res_incomp)
                ratio = np.max(
                    [np.abs(coef_in / coef_out),
                     np.abs(coef_out / coef_in)])
                
            else:
                coef_in = 0
                coef_out = 0
                ratio = 0.


        elif self.loss == 'rmse':
            #use_M-regression (huber loss)
            use_huber=True
            if use_huber:
                q_alpha=0.5
                q_in=np.percentile(np.abs(y[comp_pts] - curr_totals[comp_pts]),q_alpha)
                res_in=y[comp_pts] - curr_totals[comp_pts]
                median_in=np.median(res_in)
                coef_in = median_in + (
                            1/len(comp_pts)*(np.sum(np.sign(res_in-
                                 median_in)*np.min(np.hstack([q_in*np.ones(len(res_in)).reshape([-1,1]),np.abs(res_in-
                                 median_in).reshape([-1,1])]),axis=1))))

   
                if self.learner_type_code == 1:
                    coef_out=0
                else:
                    q_out=np.percentile(np.abs(y[incomp_pts] - curr_totals[incomp_pts]),q_alpha)
                    res_out=y[incomp_pts] - curr_totals[incomp_pts]
                    median_out=np.median(res_out) 
                    coef_out = median_out + (
                            1/len(incomp_pts)*(np.sum(np.sign(res_out-
                                 median_out)*np.min(np.hstack([q_out*np.ones(len(res_out)).reshape([-1,1]),np.abs(res_out-
                                 median_out).reshape([-1,1])]),axis=1))))
            else:
                coef_in = np.median(
                    y[comp_pts] - curr_totals[comp_pts])                                
                coef_out = (0 if self.learner_type_code == 1 else
                        np.median(y[incomp_pts] -
                                  curr_totals[incomp_pts])) 


        # calc loss
        new_preds=curr_ttls.copy()
        new_preds[comp_pts]+=coef_in
        new_preds[incomp_pts]+=coef_out
        loss_=self.calc_loss(new_preds,y)
        
        # one-sided check
        if self.learner_type_code == 1: 
            coef_out=0.
            
        return [coef_out,coef_in,ratio,loss_]
    def calc_loss(self,curr_ttls,y):
        if self.loss=='deviance':
            return calc_loss_deviance(curr_ttls,y)
        else:
            return calc_loss_rmse(curr_ttls,y)
    def fit_from_cache(
            self,
            cached_local_hp_data,
            X,
            y,
            res_train,
            curr_totals,
            hp_reg=None,
            hp_reg_c=None):
        curr_loss=self.calc_loss(curr_totals,y)
        best = [curr_loss, -1, -99, -1, [-1, -1]]    # err, base, dirn, hp, coefs
        for i in np.arange(X.shape[0]):
            data_i = cached_local_hp_data[i]
            for dirn in [-1, +1]:
                data_dirn = data_i[dirn]
                vs = data_dirn['vs']
                hps = data_dirn['hps']
                comp_idxs = data_dirn['comp_idxs']
                incomp_idxs = data_dirn['incomp_idxs']
                for i_v in np.arange(len(vs)):
                    comp_pts = comp_idxs[i_v]
                    incomp_pts = incomp_idxs[i_v]#np.asarray(np.setdiff1d(
                        #np.arange(X.shape[0]), comp_pts))
                    hp = hps[i_v, :]
#                    res_comp_pts = res_train[comp_pts]
#                    res_incomp_pts = res_train[incomp_pts]
#                    mean_res_in = np.mean(res_comp_pts)
#                    mean_res_out = np.mean(res_incomp_pts)
#                    sse = np.sum((res_train[comp_pts] - mean_res_in)**2) + \
#                        np.sum((res_train[incomp_pts] - mean_res_out)**2)
#                    if hp_reg is not None and len(self.nmt_feats) > 0:
#                        if hp_reg == 'L1_nmt' or hp_reg == 'L2_nmt':
#                            sse = sse + hp_reg_c * \
#                                np.linalg.norm(hp[self.nmt_feats - 1], ord=1
#                                               if hp_reg == 'L1_nmt' else
#                                               2)**(1 if hp_reg == 'L1_nmt'
#                                                    else 2)
#                        elif hp_reg == 'L1' or hp_reg == 'L2':
#                            sse = sse + hp_reg_c * \
#                                np.linalg.norm(hp, ord=1 if hp_reg == 'L1' else
#                                               2)**(1 if hp_reg == 'L1' else 2)
#                    if sse <= best[0] and len(
#                            comp_pts) > 0:
#                        if self.loss == 'deviance':
#                            
#                            sum_res_comp = np.sum(np.abs(res_comp_pts) * (
#                                1 - np.abs(res_comp_pts)))
#                            sum_res_incomp = np.sum(np.abs(res_incomp_pts) * (
#                                1 - np.abs(res_incomp_pts)))
#                            signed_sum_res_comp = np.sum(res_comp_pts)
#                            signed_sum_res_incomp = np.sum(res_incomp_pts)
#                            if (sum_res_comp > 1e-9 and
#                                    sum_res_incomp > 1e-9 and
#                                    np.abs(signed_sum_res_comp) > 1e-9 and
#                                    np.abs(signed_sum_res_incomp) > 1e-9):
#                                coef_in = 0.5 * signed_sum_res_comp / \
#                                    (sum_res_comp)
#                                if self.learner_type_code == 0:  # two sided
#                                    coef_out = 0.5 * signed_sum_res_incomp / \
#                                        (sum_res_incomp)
#                                    ratio = np.max(
#                                        [np.abs(coef_in / coef_out),
#                                         np.abs(coef_out / coef_in)])
#                                elif self.learner_type_code == 1:  # one-sided
#                                    [coef_out, ratio] = [0., 0.5]
#                            else:
#                                coef_in = 0
#                                coef_out = 0
#                                ratio = 0.
#                        elif self.loss == 'rmse':
#                            #use_M-regression (huber loss)
#                            use_huber=True
#                            if use_huber:
#                                q_alpha=0.5
#                                q_in=np.percentile(np.abs(y[comp_pts] - curr_totals[comp_pts]),q_alpha)
#                                res_in=y[comp_pts] - curr_totals[comp_pts]
#                                median_in=np.median(res_in)
#                                coef_in = median_in + (
#                                            1/len(comp_pts)*(np.sum(np.sign(res_in-
#                                                 median_in)*np.min(np.hstack([q_in*np.ones(len(res_in)).reshape([-1,1]),np.abs(res_in-
#                                                 median_in).reshape([-1,1])]),axis=1))))
#        
#           
#                                if self.learner_type_code == 1:
#                                    coef_out=0
#                                else:
#                                    q_out=np.percentile(np.abs(y[incomp_pts] - curr_totals[incomp_pts]),q_alpha)
#                                    res_out=y[incomp_pts] - curr_totals[incomp_pts]
#                                    median_out=np.median(res_out) 
#                                    coef_out = median_out + (
#                                            1/len(incomp_pts)*(np.sum(np.sign(res_out-
#                                                 median_out)*np.min(np.hstack([q_out*np.ones(len(res_out)).reshape([-1,1]),np.abs(res_out-
#                                                 median_out).reshape([-1,1])]),axis=1))))
#                            else:
#                                coef_in = np.median(
#                                    y[comp_pts] - curr_totals[comp_pts])                                
#                                coef_out = (0 if self.learner_type_code == 1 else
#                                        np.median(y[incomp_pts] -
#                                                  curr_totals[incomp_pts])) 
#                            ratio = 0.
                    [coef_out,coef_in,ratio,loss_]=self.calc_coefs(comp_pts,hp,res_train,y,curr_totals)
                    if loss_<=best[0]: 
                        #if np.sign(coef_in) == dirn and np.sign(coef_out) == -dirn (
                        if coef_in*dirn > coef_out * dirn and (
                                coef_in != np.inf and coef_out != np.inf and
                                ratio < 1e9):
                            best = [
                                loss_, i, dirn, hp, [
                                    coef_out, coef_in]]
        self.x_base = X[best[1], :]
        self.coefs = best[4]
        self.dirn = best[2]
        self.nmt_hyperplane = best[3]
        return self

    def transform(self, X_pred_):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        res = np.asarray([1 if self.comparator.compare(self.x_base, X_pred_[
            i, :]) * self.dirn in [0, 1] else 0 for i in
            np.arange(X_pred_.shape[0])])

        return res


class MonoBoost():
    """ Partially Monotone Boosting classifier
    var

   Attributes
    ----------
    eg_attr: list of DecisionTreeClassifier
        The collection of fitted sub-estimators.


    References
    ----------

    XXX

    See also
    --------
    YYY
    """
    # NB: fit_algo is irrelevant for fit_type='quadratic'

    def __init__(self,
                 n_feats,
                 incr_feats,
                 decr_feats,
                 num_estimators=10,
                 fit_algo='L2-one-class',
                 eta=1.,
                 vs=[0.001,
                     0.1,
                     0.25,
                     0.5,
                     1],
                 verbose=False,
                 hp_reg=None,
                 hp_reg_c=None,
                 incomp_pred_type='default',
                 learner_type='one-sided',
                 random_state=None,
                 standardise=True,
                 classes=None,
                 loss='auto'):
        self.X = None
        self.y = None
        self.classes=classes
        self.n_feats = n_feats
        self.incr_feats = np.asarray(incr_feats)
        self.decr_feats = np.asarray(decr_feats)
        self.mt_feats = np.asarray(list(incr_feats) + list(decr_feats))
        self.nmt_feats = np.asarray(
            [j for j in np.arange(n_feats) + 1 if j not in self.mt_feats])
        self.mt_feat_types=np.zeros(n_feats,dtype=np.float64)
        if len(self.incr_feats)>0:
            self.mt_feat_types[self.incr_feats-1]=+1.
        if len(self.decr_feats)>0:
            self.mt_feat_types[self.decr_feats-1]=-1.
        self.fitted = False
        self.standardise = standardise
        self.fit_algo = fit_algo
        self.eta = eta
        self.num_estimators = num_estimators
        self.vs = vs
        self.mt_comparator = MonoComparator(
            n_feats, incr_feats, decr_feats, nmt_hyperplane=None)
        self.verbose = verbose
        self.hp_reg = hp_reg
        self.hp_reg_c = hp_reg_c
        self.y_pred_num_comp_ = None
        self.incomp_pred_type = incomp_pred_type
        self.learner_type = learner_type
        self.random_state = np.random.randint(
            1e6) if random_state is None else random_state
        np.random.seed(self.random_state)
        self.loss = loss#'auto'
        self.__estimators_base_pts__=None
        self.__estimators_dirns__=None
        self.__estimators_intercepts__=None
        self.__estimators_hyperplanes__=None
     
    def get_estimator_matrices(self):
        if self.__estimators_base_pts__ is None:
            self.__estimators_base_pts__={}
            self.__estimators_dirns__={}
            self.__estimators_hyperplanes__={}
            self.__estimators_intercepts__={}
            self.__estimators_coefs__={}
            for k in self.ks:
                self.__estimators_base_pts__[k]=np.asarray([est.x_base for est in self.estimators[k]],dtype=np.float64)
                self.__estimators_dirns__[k]=np.asarray([est.dirn for est in self.estimators[k]],dtype=np.float64)
                self.__estimators_hyperplanes__[k]=np.asarray([est.nmt_hyperplane for est in self.estimators[k]],dtype=np.float64)
                self.__estimators_intercepts__[k]=np.asarray([est.intercept_ for est in self.estimators[k]],dtype=np.float64)
                self.__estimators_coefs__[k]=np.asarray([est.coefs for est in self.estimators[k]],dtype=np.float64)
            
        return [self.__estimators_base_pts__,
                self.__estimators_dirns__,
                self.__estimators_intercepts__,
                self.__estimators_hyperplanes__,
                self.__estimators_coefs__]
    @property
    def y_maj_class_calc(self):
        """I'm the 'x' property."""
        return -1 if np.sum(self.y == -1) / len(self.y) >= 0.5 else +1

    @property
    def y_pred_num_comp(self):
        """I'm the 'x' property."""
        if not hasattr(self, 'y_pred_num_comp_'):
            self.y_pred_num_comp_ = None
        if self.y_pred_num_comp_ is None:
            [ypred, num_comp] = self.predict_proba(self.X)
            self.y_pred_num_comp_ = num_comp
        return self.y_pred_num_comp_

    def solve_hp(self, incr_feats, decr_feats, delta_X, v, weights=None):
        mt_feat_types_=np.zeros(delta_X.shape[1])
        if len(incr_feats)>0:
            mt_feat_types_[incr_feats-1]=1
        if len(decr_feats)>0:
            mt_feat_types_[decr_feats-1]=-1
        
        w=fit_one_class_svm(delta_X,weights,v,mt_feat_types_)
        return w
#        
#        N = delta_X.shape[0]
#        p = delta_X.shape[1]
#        num_feats = p
#        mt_feats = np.asarray(list(incr_feats) + list(decr_feats))
#        nmt_feats = np.asarray(
#            [j for j in np.arange(num_feats) + 1 if j not in mt_feats])
#        solvers.options['show_progress'] = False
#        if N == 0:
#            return [-99]
#        else:
#            # Build QP matrices
#            # Minimize     1/2 x^T P x + q^T x
#            # Subject to   G x <= h
#            #             A x = b
#            if weights is None:
#                weights = np.ones(N)
#            P = np.zeros([p + N, p + N])
#            for ip in nmt_feats - 1:
#                P[ip, ip] = 1
#            q = 1 / (N * v) * np.ones((N + p, 1))
#            q[0:p, 0] = 0
#            q[p:, 0] = q[p:, 0] * weights
#            G1a = np.zeros([p, p])
#            for ip in np.arange(p):
#                G1a[ip, ip] = -1 if ip in mt_feats - 1 else 1
#            G1 = np.hstack([G1a, np.zeros([p, N])])
#            G2 = np.hstack([np.zeros([N, p]), -np.eye(N)])
#            G3 = np.hstack([delta_X, -np.eye(N)])
#            G = np.vstack([G1, G2, G3])
#            h = np.zeros([p + 2 * N])
#            A = np.zeros([1, p + N])
#            for ip in np.arange(p):
#                A[0, ip] = 1 if ip in mt_feats - 1 else -1
#            b = np.asarray([1.])
#            P = cvxmat(P)
#            q = cvxmat(q)
#            A = cvxmat(A)
#            b = cvxmat(b)
#            # options['abstol']=1e-20 #(default: 1e-7).
#            # options['reltol']=1e-11 #(default: 1e-6)
#            sol = solvers.qp(P, q, cvxmat(G), cvxmat(h), A, b)
#            if sol['status'] != 'optimal':
#                print(
#                    '****** NOT OPTIMAL ' +
#                    sol['status'] +
#                    ' ******* [N=' +
#                    str(N) +
#                    ', p=' +
#                    str(p) +
#                    ']')
#                return [-99]
#            else:
#                soln = sol['x']
#                w = np.ravel(soln[0:p, :])
#                # err = np.asarray(soln[-N:, :])
#                return w


    def get_deltas(self, X_base_pt, X, y):
        dirns = [-1, 1]
        deltas = [np.zeros([X.shape[0], X.shape[1]]),
                  np.zeros([X.shape[0], X.shape[1]])]
        comp_indxs = []
        for dirn in dirns:
            idirn = 0 if dirn == -1 else 1
            i_j = 0
            comp_indxs_ = []
            for j in np.arange(X.shape[0]):
                if y[j] == -dirn:
                    comp = self.mt_comparator.compare(X_base_pt, X[j, :])
                    if comp == dirn or comp == 0:
                        comp_indxs_ = comp_indxs_ + [j]
                        d_ = X[j, :] - X_base_pt
                        # if not np.all(d_==0.):
                        deltas[idirn][i_j, :] = np.abs(d_)
                        i_j = i_j + 1
            deltas[idirn] = deltas[idirn][0:i_j, :]
            comp_indxs = comp_indxs + [np.asarray(comp_indxs_)]
        return [comp_indxs, deltas]

    def get_comparable_points(self,X,X_base_pts,nmt_hps,intercepts,dirn):
        if np.isscalar(intercepts):
            intercepts=np.asarray([intercepts],dtype=np.float64)

        if len(X.shape)<2:
            X_=np.asarray(X.reshape([1,-1]),dtype=np.float64)
        else:
            X_=np.asarray(X,dtype=np.float64)
        if len(X_base_pts.shape)<2:
            X_base_pts_=np.asarray(X_base_pts.reshape([1,-1]),dtype=np.float64)
        else:
            X_base_pts_=np.asarray(X_base_pts,dtype=np.float64)
        if len(nmt_hps.shape)<2:
            nmt_hps_=np.asarray(nmt_hps.reshape([1,-1]),dtype=np.float64)
        else:
            nmt_hps_=np.asarray(nmt_hps,dtype=np.float64)     
        X_comp_pts=np.zeros([X_.shape[0],X_base_pts_.shape[0]],dtype=np.int32)
        monoboost.apply_rules_c(X_,
                  X_base_pts_, 
                  nmt_hps_,
                  intercepts,
                  self.mt_feat_types,
                  np.float64(dirn),
                  0,
                   X_comp_pts)
        return X_comp_pts
        
    def get_base_comparable_pairs(self,X):
        if len(X.shape)==1:
            X_=X.reshape([1,-1])
        else:
            X_=X

        # set nmt to zero because we are not considering nmt feats
        __hyperplanes__=np.zeros(X_.shape,dtype=np.float64)
        __intercepts__=np.zeros(X_.shape[0],dtype=np.float64)

        X_pts_above=np.zeros([X_.shape[0],X_.shape[0]],dtype=np.int32)
        monoboost.apply_rules_c(X_,
                  X_, 
                  __hyperplanes__,
                  __intercepts__,
                  self.mt_feat_types,
                  np.float64(-1),
                  0,
                   X_pts_above)

        X_pts_below=np.zeros([X_.shape[0],X_.shape[0]],dtype=np.int32)
        monoboost.apply_rules_c(X_,
                  X_, 
                  __hyperplanes__,
                  __intercepts__,
                  self.mt_feat_types,
                  np.float64(+1),
                  0,
                   X_pts_below)
        
        # remove self-comparisons
#        for i in np.arange(X_.shape[0]):
#            X_pts_above[i,i]=0
#            X_pts_below[i,i]=0
        return [X_pts_below==1,X_pts_above==1]
    
    def fit_cache(self, X, y, svm_vs):
        X_rev_dec = X.copy()
        for dec_feat in self.decr_feats:
            X_rev_dec[:, dec_feat - 1] = -1 * X_rev_dec[:, dec_feat - 1]
        hp_data = dict()
        calc_hyperplanes = self.learner_type != 'ignore_nmt_feats'
        # get comparable pts
        [X_pts_below,X_pts_above]=self.get_base_comparable_pairs(X)
        for i in np.arange(X.shape[0]):
            dirn_ = dict()
            dirn_pos = dict()
            dirn_neg = dict()
            dirn_[-1] = dirn_neg
            dirn_[1] = dirn_pos
            hp_data[i] = dirn_
            x_i = X[i, :]
            # get base comparable pts in given direction
#            pts_above = []
#            pts_below = []
#            for j in np.arange(X.shape[0]):
#                # if i!=j:
#                comp = self.mt_comparator.compare(X[i, :], X[j, :])
#                if comp == +1 or comp == 0:
#                    pts_above = pts_above + [j]
#                if comp == -1 or comp == 0:
#                    pts_below = pts_below + [j]
            pts_above=np.arange(X.shape[0])[X_pts_above[i,:]]
            pts_below=np.arange(X.shape[0])[X_pts_below[i,:]]
            if self.fit_algo=='none': # ignore nmt feats
                for dirn in [-1, +1]:
                    base_comp_idxs = np.asarray(
                        pts_above if dirn == 1 else pts_below)
                    base_incomp_idxs = np.asarray(np.setdiff1d(
                            np.arange(X.shape[0]), base_comp_idxs))
                    hps = np.zeros([1, X.shape[1]])
                    
                    hp_data[i][dirn]['hps'] = hps#None #hps
                    hp_data[i][dirn]['vs'] = [99]#None #np.asarray(vs)
                    hp_data[i][dirn]['comp_idxs'] = [base_comp_idxs] #comp_idxs
                    hp_data[i][dirn]['incomp_idxs'] = [base_incomp_idxs] #incomp_idxs
            else: #calc hyperplanes
                # calculate hyperplanes
                for dirn in [-1, +1]:
                    base_comp_idxs = np.asarray(
                        pts_above if dirn == 1 else pts_below)
                    base_incomp_idxs = np.asarray(np.setdiff1d(
                            np.arange(X.shape[0]), base_comp_idxs))
                    hps = np.zeros([0, X.shape[1]])
                    if calc_hyperplanes:
                        if len(base_comp_idxs) > 0:
                            y_signed = y[base_comp_idxs]
                            if len(y_signed[y_signed * dirn < 0]) >= 2 and len(
                                    self.nmt_feats) != 0:  # fit svm for each v
                                hps = np.zeros([len(svm_vs), X.shape[1]])
                                comp_idxs = []
                                incomp_idxs=[]
                                vs = []
                                deltas = np.zeros(
                                    [len(base_comp_idxs), X.shape[1]])
                                weights = np.zeros(X.shape[0])
                                i_j = 0
                                for j in base_comp_idxs:
                                    if np.sign(y[j]) == -dirn and j != i:
                                        deltas[i_j, :] = dirn * \
                                            (X_rev_dec[j, :] - X_rev_dec[i, :])
                                        weights[i_j] = np.abs(y[j])
                                        for k in self.nmt_feats:
                                            deltas[i_j, k -
                                                   1] = np.abs(deltas[i_j, k - 1])
                                        i_j = i_j + 1
                                deltas = deltas[0:i_j, :]
                                weights = weights[0:i_j]
                                i_v_real = 0
                                smt_comparator = MonoComparator(
                                    self.n_feats, self.incr_feats, self.decr_feats,
                                    nmt_hyperplane=None)
                                for i_v in np.arange(len(svm_vs) - 1, -1, -1):
                                    v = svm_vs[i_v]
                                    fitted_hp = self.solve_hp(
                                        self.incr_feats, self.decr_feats, deltas,
                                        v, weights)
                                    if fitted_hp[0] != -99 and np.sum(
                                            np.abs(fitted_hp - hps[i_v_real - 1, :]
                                                   )) > 5e-4:
                                        smt_comparator.nmt_hyperplane = fitted_hp
                                        comp_pts_v = []
                                        if len(comp_idxs) == 0:
                                            com_inds = base_comp_idxs
                                        else:
                                            com_inds = comp_idxs[-1]
                                            
    #                                    for c in com_inds:
    #                                        comp = smt_comparator.compare(
    #                                            x_i, X[c, :])
    #                                        if comp == dirn or comp == 0:
    #                                            comp_pts_v = comp_pts_v + [c]
    #                                    
                                        comp_pts_mask=self.get_comparable_points(X[com_inds, :],x_i,fitted_hp,0.,dirn)
                                        comp_pts_v=com_inds[comp_pts_mask[:,0]==1]
                                        incomp_pts_v=com_inds[comp_pts_mask[:,0]==0]
                                        
                                        hps[i_v_real, :] = fitted_hp
                                        comp_idxs = comp_idxs + \
                                            [np.asarray(comp_pts_v)]
                                        incomp_idxs = incomp_idxs + \
                                            [np.asarray(incomp_pts_v)]
                                        vs = vs + [v]
                                        i_v_real = i_v_real + 1
                                hps = hps[0:i_v_real, :]
                    # we have no hyperplanes, add the default (NMT feats are
                    # irrelevant)
                    if hps.shape[0] == 0:
                        hps = np.zeros([1, X.shape[1]])
                        hps[0, :] = np.asarray(
                            [1 if kk in self.mt_feats else 0 for kk in
                             np.arange(X.shape[1]) + 1]) / len(self.mt_feats)
                        vs = [-99]
                        comp_idxs = [base_comp_idxs]
                        incomp_idxs = [base_incomp_idxs]
    
                    hp_data[i][dirn]['hps'] = hps
                    hp_data[i][dirn]['vs'] = np.asarray(vs)
                    hp_data[i][dirn]['comp_idxs'] = comp_idxs
                    hp_data[i][dirn]['incomp_idxs'] = incomp_idxs
            hp_data[i][-1]['base_comp_idxs'] = np.asarray(pts_below)
            hp_data[i][1]['base_comp_idxs'] = np.asarray(pts_above)
            
        return hp_data
    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        # self.class_weight_ = compute_class_weight(self.class_weight, cls, y_)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                % len(cls))

        self.classes_ = cls
        self.n_classes_ = len(cls)
        return np.asarray(y, dtype=np.float64, order='C')
    
    def fit(self, X, y):
        """Fits one hyperplane per non-monotone feature
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        feat_softening_angles : array-like, shape = [m]
            an angle for each feature, ignored for mt feats, and used as
            softening angle otherwise (in degrees)
        Returns
        -------
        self : object
            Returns self.
        """
        solvers.options['show_progress'] = False
        self.__estimators_base_pts__=None
        # solvers.options['feastol'] = 1e-16# default: 1e-7
        # save vars
        if self.classes is None:
            self.classes = np.sort(np.unique(y))
        
        self.n_classes_ = len(self.classes)
        if self.loss == 'auto':
            self.loss_ = 'deviance' if self.n_classes_ <= 10 else 'rmse' # and np.all(
                #self.classes == np.asarray([-1, 1])) else 'rmse'
        else:
            self.loss_ = self.loss
        if self.loss_ == 'deviance':    
            y_indexed = self._validate_targets(y)
            self.y=np.zeros([self.n_classes_-1,len(y)])
            self.ks=np.arange(self.n_classes_-1)
        else:
            self.y=y
            self.ks=[0] 
        
            
        self.X = X
        
        if self.standardise:
            self.scale = Scale()
            self.scale.train(X)
            self.X_scaled = self.scale.scale(X)
        else:
            self.X_scaled = X
        self.y_pred_num_comp_ = None
        self.n_datapts = X.shape[0]


        self.hp_cache=[]
        self.intercept_=[0. for i in np.arange(self.n_classes_-1)]
        self.estimators = {}
        self.train_err_all = {}
        self.y_pred_train_all={}
        self.y_pred={}
        # Train base models (binary RFs monotone ensembled)
        for k in self.ks:
            if self.loss_=='deviance':
                y_ = np.array(y_indexed > k, dtype=np.float64)
                y_[y_==0]=-1
                self.y[k,:]=y_
            else:
                y_ = y
            
            # get cache of possible hyperplanes & comparable points
            self.hp_cache = self.hp_cache + [self.fit_cache(self.X_scaled, y_, self.vs)]
            # start boosting!
            cont = True
            i_iter = 0
            y_std = y_.copy()
            if self.loss_ == 'deviance':
                y_std[y_std == -1] = 0
                prob_class_one = np.sum(y_std == 1) / len(y_std)
                # Friedman 2001 equation (29) and Elements of Statistical Learning
                # Algorithm 10.3 Line 1: for binary classification we can do better
                # than initialising to 0 as in Algo 10.4
                self.intercept_[k] = 0.5 * \
                    (np.log(prob_class_one) - np.log(1 - prob_class_one))
                res_train = y_std - prob_class_one
            elif self.loss_ == 'rmse':
                self.intercept_[k] = np.median(y_std)
                res_train = robust_sign(y_std - self.intercept_[k])

            curr_ttls = np.zeros(X.shape[0]) + self.intercept_[k]
            self.estimators[k] = []
            self.train_err_all[k] = np.zeros(self.num_estimators)
            self.y_pred_train_all[k] = np.zeros([X.shape[0], self.num_estimators])
    
            while cont:
                # find next best rule
                est = MonoLearner(
                    self.n_feats,
                    self.incr_feats,
                    self.decr_feats,
                    learner_type=self.learner_type,
                    loss=self.loss_)
                est.fit_from_cache(
                    self.hp_cache[k],
                    self.X_scaled,
                    y_std,
                    res_train,
                    curr_ttls,
                    self.hp_reg,
                    self.hp_reg_c)
                if est.dirn == -99:
                    if len(self.estimators[k]) < self.num_estimators:
                        self.num_estimators = len(self.estimators[k])
                        print('Only made it to ' + str(len(self.estimators[k])) +
                              ' iterations and couldnt find more viable splits')
                    cont = False
    
                else:
                    #pred_, is_comp] = est.predict_proba(self.X_scaled)
                    [pred_, is_comp] = est.decision_function(self.X_scaled)
                    curr_ttls = curr_ttls + self.eta * pred_
                    if self.standardise:  # unstandardise the estimator
                        est.x_base = self.scale.unscale(est.x_base)
                        est.nmt_hyperplane = self.scale.scale(est.nmt_hyperplane)
                    self.estimators[k] = self.estimators[k] + [est]
    
                    # calc next iter
                    if self.loss_ == 'deviance':
                        # prevents overflow at the next step
                        curr_ttls[curr_ttls > 100] = 100
                        p_ = np.exp(curr_ttls) / \
                            (np.exp(curr_ttls) + np.exp(-curr_ttls))
                        res_train = y_std - p_
                        self.y_pred_train_all[k][:, i_iter] = np.sign(
                            curr_ttls + 1e-9)
                        self.train_err_all[k][i_iter] = np.sum(
                            self.y_pred_train_all[k][:, i_iter] != y_) / len(y_)
                        if self.verbose:
                            print(
                                np.sum(self.y_pred_train_all[k][:, i_iter] != y_
                                       ) / len(y_))
                    elif self.loss_ == 'rmse':
                        res_train = robust_sign(y_std - curr_ttls)
                    cont = i_iter < (self.num_estimators - 1)
                    i_iter = i_iter + 1
            if self.loss_ == 'deviance':
                self.y_pred[k] = np.sign(curr_ttls)
                self.y_pred[k][self.y_pred == 0] = -1
            elif self.loss_ == 'rmse':
                self.y_pred[k] = curr_ttls
        return

#    def predict_proba(self, X_pred, cum=False):
#        """Predict class or regression value for X.
#        For a classification model, the predicted class for each sample in X is
#        returned. For a regression model, the predicted value based on X is
#        returned.
#        Parameters
#        ----------
#        X : array-like or sparse matrix of shape = [n_samples, n_features]
#            The input samples. Internally, it will be converted to
#            ``dtype=np.float32`` and if a sparse matrix is provided
#            to a sparse ``csr_matrix``.
#        cum : boolean, (default=False)
#            True to include predictions for all stages cumulatively.
#        Returns
#        -------
#        y : array of shape = [n_samples] or [n_samples, n_outputs]
#            The predicted classes, or the predict values.
#        """
#        if len(X_pred.shape) == 0:
#            X_pred_ = np.zeros([1, len(X_pred)])
#            X_pred_[0, :] = X_pred
#        else:
#            X_pred_ = X_pred
#        res = np.zeros([X_pred.shape[0], len(self.estimators)]
#                       ) if cum else np.zeros(X_pred.shape[0])
#        res = res + self.intercept_ 
#        num_comp = np.zeros([X_pred.shape[0], len(self.estimators)]
#                            ) if cum else np.zeros(X_pred.shape[0])
#        for i in np.arange(len(self.estimators)):
#            [pred_, is_comp] = self.estimators[i].predict_proba(X_pred)
#            if cum:
#                res[:, i] = res[:, i - 1] + self.eta * pred_
#                num_comp[:, i] = num_comp[:, i - 1] + is_comp
#            else:
#                res = res + self.eta * pred_
#                num_comp = num_comp + is_comp
#
#        return [res, num_comp]

    def decision_function(self, X_pred, cum=False):
        if len(X_pred.shape)==1:
            X_pred=X_pred.reshape([1,-1])
        # new version
        [__estimators_base_pts__,
        __estimators_dirns__,
        __estimators_intercepts__,
        __estimators_hyperplanes__,
        __estimators_coefs__]=self.get_estimator_matrices()
        
        dec_fn={}#np.zeros([X_pred.shape[0],self.n_classes_-1],dtype=np.float64)
        num_comp={}#np.zeros([X_pred.shape[0],self.n_classes_-1],dtype=np.float64)
        for k in self.ks:
            #__estimators_hyperplanes__=np.zeros(__estimators_hyperplanes__.shape,dtype=np.float64)
            X_rule_transform=np.zeros([X_pred.shape[0],__estimators_base_pts__[k].shape[0]],dtype=np.int32)
            X_coef_contribs=np.zeros([X_pred.shape[0],__estimators_base_pts__[k].shape[0]],dtype=np.float64)
            for dirn in [-1,+1]:
                dirn_rule_mask=__estimators_dirns__[k]==dirn
                X_rule_transform_=np.zeros([X_pred.shape[0],np.sum(dirn_rule_mask)],dtype=np.int32)
            
                monoboost.apply_rules_c(X_pred,
                      __estimators_base_pts__[k][dirn_rule_mask,:], 
                      __estimators_hyperplanes__[k][dirn_rule_mask,:],
                      __estimators_intercepts__[k][dirn_rule_mask],
                      self.mt_feat_types,
                      np.float64(dirn),
                      0,
                       X_rule_transform_)
                X_rule_transform[:,dirn_rule_mask]=X_rule_transform_
                X_coef_contribs[:,dirn_rule_mask]=X_coef_contribs[:,dirn_rule_mask]+self.eta*__estimators_coefs__[k][dirn_rule_mask,1]*X_rule_transform_
                X_coef_contribs[:,dirn_rule_mask]=X_coef_contribs[:,dirn_rule_mask]+self.eta*__estimators_coefs__[k][dirn_rule_mask,0]*np.abs(X_rule_transform_-1)
            if cum:
                n_est=self.num_estimators #len(self.estimators[k])
                dec_fn[k] = np.zeros([X_pred.shape[0], n_est])+self.intercept_[k]
                num_comp[k]= np.zeros([X_pred.shape[0], n_est])
                
                for j in np.arange(n_est):
                    for jj in np.arange(j,n_est):
                        dec_fn[k][:,jj]=dec_fn[k][:,jj]+X_coef_contribs[:,j]
                        num_comp[k][:,jj]=num_comp[k][:,jj]+X_rule_transform[:,j]
            else:
                dec_fn[k]=np.sum(X_coef_contribs,axis=1)+self.intercept_[k]
                num_comp[k]=np.sum(X_rule_transform,axis=1)
        return [dec_fn,num_comp]
        
    def predict(self, X_pred, cum=False):
        if len(X_pred.shape)==1:
            X_pred=X_pred.reshape([1,-1])

        #[pred, comp] = self.predict_proba(X_pred, cum)
        [dec, comp]=self.decision_function(X_pred, cum)
#        if cum:
#            predictions=np.zeros([X_pred.shape[0],dec[0].shape[1]],dtype=np.int32)
#            for k in np.arange(self.n_classes_-1):
#                predictions += dec[k]>=0
#            
#                
#        else:
        self._last_comp=comp
        if cum:
            predictions=np.zeros([dec[0].shape[0],self.num_estimators],dtype=np.int32)
        else:
            predictions=np.zeros(X_pred.shape[0],dtype=np.int32)
        if self.loss_=='deviance':
            for k in self.ks:
                predictions += dec[k]>=0 
            return self.classes_.take(np.asarray(predictions, dtype=np.intp))
        else:
            predictions=dec[0]
            return predictions
        
        #return np.sign(pred) if self.loss_ == 'deviance' else pred

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        res_old= np.array([instance.transform(X)
                         for instance in self.estimators]).T
             
        return   res_old                                          


#class MonoBoostEnsemble():
#    """ Partially Monotone Boosting classifier ensemble
#    var
#
#   Attributes
#    ----------
#    eg_attr: list of DecisionTreeClassifier
#        The collection of fitted sub-estimators.
#
#
#    References
#    ----------
#
#    XXX
#
#    See also
#    --------
#    YYY
#    """
#    # NB: fit_algo is irrelevant for fit_type='quadratic'
#
#    def __init__(self,
#                 n_feats,
#                 incr_feats,
#                 decr_feats,
#                 num_estimators=100,
#                 fit_algo='L2-one-class',
#                 eta=1.,
#                 vs=[0.001,
#                     0.1,
#                     0.25,
#                     0.5,
#                     1],
#                 verbose=False,
#                 learner_incomp_pred_type='default',
#                 learner_type='one-sided',
#                 learner_num_estimators=10,
#                 learner_eta=1.0,
#                 learner_v_mode='random',
#                 sample_fract=1.0,
#                 random_state=None,
#                 standardise=True,
#                 classes=None):
#        self.X = None
#        self.y = None
#        self.classes=classes
#        self.n_feats = n_feats
#        self.incr_feats = np.asarray(incr_feats)
#        self.decr_feats = np.asarray(decr_feats)
#        self.mt_feats = np.asarray(list(incr_feats) + list(decr_feats))
#        self.nmt_feats = np.asarray(
#            [j for j in np.arange(n_feats) + 1 if j not in self.mt_feats])
#        self.fitted = False
#        self.fit_algo = fit_algo
#        self.eta = eta
#        self.num_estimators = num_estimators
#        self.vs = vs
#        self.verbose = verbose
#        self.y_pred_num_comp_ = None
#        self.learner_incomp_pred_type = learner_incomp_pred_type
#        self.learner_type = learner_type
#        self.learner_num_estimators = learner_num_estimators
#        self.learner_eta = learner_eta
#        self.learner_v_mode = learner_v_mode
#        self.sample_fract = sample_fract
#        self.random_state = np.random.int(
#            1e6) if random_state is None else random_state
#        self.standardise = standardise
#        np.random.seed(self.random_state)
#        self.loss = 'auto'
#
#    @property
#    def y_maj_class_calc(self):
#        """I'm the 'x' property."""
#        maj_classes=np.zeros(len(self.ks))
#        for k in self.ks:
#            maj_classes[k]=-1 if np.sum(self.y_ensembled[:,k] == -1) / self.y_ensembled.shape[0] >= 0.5 else +1 
#        return maj_classes #-1 if np.sum(self.y == -1) / len(self.y) >= 0.5 else +1
#
#    def _validate_targets(self, y):
#        y_ = column_or_1d(y, warn=True)
#        check_classification_targets(y)
#        cls, y = np.unique(y_, return_inverse=True)
#        # self.class_weight_ = compute_class_weight(self.class_weight, cls, y_)
#        if len(cls) < 2:
#            raise ValueError(
#                "The number of classes has to be greater than one; got %d"
#                % len(cls))
#
#        self.classes_ = cls
#        self.n_classes_ = len(cls)
#        return np.asarray(y, dtype=np.float64, order='C')
#    
#    def fit(self, X, y):
#        """Fits one hyperplane per non-monotone feature
#        Parameters
#        ----------
#        X : array-like or sparse matrix of shape = [n_samples, n_features]
#            The training input samples. Internally, its dtype will be converted
#            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
#            converted into a sparse ``csc_matrix``.
#        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
#            The target values (class labels in classification, real numbers in
#            regression).
#        feat_softening_angles : array-like, shape = [m]
#            an angle for each feature, ignored for mt feats, and used as
#            softening angle otherwise (in degrees)
#        Returns
#        -------
#        self : object
#            Returns self.
#        """
#        # save vars
#        if self.classes is None:
#            self.classes = np.sort(np.unique(y))
#        
#        self.n_classes_ = len(self.classes)
#        if self.loss == 'auto':
#            self.loss_ = 'deviance' if self.n_classes_ <= 10 else 'rmse' #and np.all(
#                #self.classes == np.asarray([-1, 1])) else 'rmse'
#        else:
#            self.loss_ = self.loss
#        if self.loss_ == 'deviance':    
#            y_indexed = self._validate_targets(y)
#            self.y=y#np.zeros([self.n_classes_-1,len(y)])
#            self.ks=np.arange(self.n_classes_-1)
##            self.y_ensembled=np.zeros([len(y),self.n_classes_-1])
##            for k in self.ks:
##                self.y_ensembled[:,k]=y_indexed>k
#        else:
#            self.y=y
#            self.ks=[0]         
#        self.X = X
#        
#        self.y_pred_num_comp_ = None
#        self.n_datapts = X.shape[0]
##        self.classes = np.sort(np.unique(y))
##        self.n_classes_ = len(self.classes)
##        if self.loss == 'auto':
##            self.loss_ = 'deviance' if self.n_classes_ <= 2 and np.all(
##                self.classes == np.asarray([-1, 1])) else 'rmse'
##        else:
##            self.loss_ = self.loss
#
#        # start boosting!
#        self.intercept_={}
#        res_train=np.zeros([X.shape[0],len(self.ks)])
#        curr_ttls=np.zeros([X.shape[0],len(self.ks)]) 
#        i_iter = 0
#        if self.loss_ == 'deviance':
#            for k in self.ks:
#                y_std = np.array(y_indexed > k, dtype=np.float64)
#                y_std[y_std==-1]=0
#                #self.y[k,:]=y_std
#                prob_class_one = np.sum(y_std == 1) / len(y_std)
#                # Friedman 2001 equation (29) and Elements of Statistical Learning
#                # Algorithm 10.3 Line 1: for binary classification we can do better
#                # than initialising to 0 as in Algo 10.4
#                self.intercept_[k] = 0.5 * \
#                    (np.log(prob_class_one) - np.log(1 - prob_class_one))
#                res_train[:,k] = y_std - prob_class_one
#                curr_ttls[:,k] =  self.intercept_[k]
#        elif self.loss_ == 'rmse':
#            y_std=y
#            self.intercept_[0] = np.median(y_std)
#            res_train[:,0] = robust_sign(y_std - self.intercept_[0])
#            curr_ttls[:,0] =  self.intercept_[0]
#        
#        
#        if self.standardise:
#            self.scale = Scale()
#            self.scale.train(X)
#            X_scaled = self.scale.scale(X)
#        else:
#            X_scaled = X
#            
#        self.estimators = {}# []
#        for k in self.ks:
#            self.estimators[k]=[]
#        cnt_estimators=0
#        while cnt_estimators < self.num_estimators-1:#len(self.estimators) <= self.num_estimators:
#            # find next best rule
#            if self.learner_v_mode == 'random':
#                vs_this = [self.vs[np.random.randint(len(self.vs))]]
#            else:
#                vs_this = self.vs
#            est = MonoBoost(
#                n_feats=self.n_feats,
#                incr_feats=self.incr_feats,
#                decr_feats=self.decr_feats,
#                num_estimators=self.learner_num_estimators,
#                fit_algo=self.fit_algo,
#                eta=self.learner_eta,
#                vs=vs_this,
#                verbose=self.verbose,
#                hp_reg=None,
#                hp_reg_c=None,
#                incomp_pred_type=self.learner_incomp_pred_type,
#                learner_type=self.learner_type,
#                random_state=self.random_state +
#                np.random.randint(1e4),
#                standardise=False,
#                classes=[-1,1],
#                loss='deviance')
#            if self.sample_fract < 1:
#                sample_indx = np.random.permutation(np.arange(X.shape[0]))[
#                    0:int(np.floor(self.sample_fract * X.shape[0]))]
#            else:
#                sample_indx = np.arange(X.shape[0])
#            X_sub = X_scaled[sample_indx, :]
#            
#            # Extract estimators - recalculate coefficients based on deviance
#            # loss
#            for k in self.ks:
#                y_std = np.array(y_indexed > k, dtype=np.float64)
#                y_in_ones=y_std.copy()
#                y_in_ones[y_in_ones==0]=-1
#                #y_std[y_std==-1]=0
#                y_std_sub=y_std[sample_indx].copy()
#                res_train_sub = res_train[sample_indx,k]
#                est.fit(X_sub, res_train_sub)
#                #for est_monolearn in est.estimators[0]:
#                #    print(est_monolearn.x_base)
#                print( ' **** Extract cones ****** [k=' + str(k) + ']' )
#                last_x_base=0.
#                for est_monolearn in est.estimators[0]:
#                    comp_pts_indx_old = np.arange(X_sub.shape[0])[np.asarray([
#                        True if est_monolearn.comparator.compare(
#                            est_monolearn.x_base, X_sub[i_, :]
#                        ) * est_monolearn.dirn in [0, 1] else False
#                        for i_ in np.arange(X_sub.shape[0])])]
#                    comp_pts_indx= np.arange(X_sub.shape[0])[est_monolearn.get_comparable_points(X_sub)]   
#                    incomp_pts_indx = np.asarray(np.setdiff1d(
#                                np.arange(X_sub.shape[0]), comp_pts_indx))
#                    if self.loss_ == 'deviance':
#                        res_comp_pts = res_train_sub[comp_pts_indx]
#                        res_incomp_pts = res_train_sub[incomp_pts_indx]
#                        coef_in =  0.5 * np.sum(res_comp_pts) / np.sum(
#                            np.abs(res_comp_pts) * (1 - np.abs(res_comp_pts)))
#                        coef_out =  0.5 * np.sum(res_incomp_pts) / np.sum(
#                            np.abs(res_incomp_pts) * (1 - np.abs(res_incomp_pts)))
#                        
#                    elif self.loss_ == 'rmse':
#                        coef_in = np.median(
#                            y_std_sub[comp_pts_indx] - curr_ttls[comp_pts_indx,k])
#                        coef_out = np.median(
#                            y_std_sub[incomp_pts_indx] - curr_ttls[incomp_pts_indx,k])
#                    #if np.sign(coef_in)*est_monolearn.dirn >0 and np.sign(coef_out)*est_monolearn.dirn <=0:
#                    if coef_in*est_monolearn.dirn > coef_out * est_monolearn.dirn:
#                        if np.linalg.norm(est_monolearn.x_base-last_x_base,2)!=0.:
#                            est_monolearn.coefs = [coef_out, coef_in]
#                        else:
#                            est_monolearn.coefs = [0, 0]
#                    else:
#                        print(' wrong sign coef - ignored ' )
#                        est_monolearn.coefs = [0, 0]
#                    last_x_base= est_monolearn.x_base
#                    # Update function totals (for ALL, not just subsample)
#                    [pred_, is_comp] = est_monolearn.decision_function(X_scaled)
#
#                    print('err before: ' + str(np.sum(np.sign(curr_ttls[:,k])!=y_in_ones)))
#                    curr_ttls[:,k] = curr_ttls[:,k] + self.eta * pred_
#                    #print('err after: ' + str(np.sum(np.sign(curr_ttls[:,k])!=y_in_ones)))
#                    if self.standardise:  # unscale this estimator
#                        est_monolearn.x_base = self.scale.unscale(
#                            est_monolearn.x_base)
#                        est_monolearn.nmt_hyperplane = self.scale.scale(
#                            est_monolearn.nmt_hyperplane) # funny but true: you need to SCALE the hyperplane
#                    self.estimators[k] = self.estimators[k] + [est_monolearn]
#                cnt_estimators=len(self.estimators[0])
#                # Update res_train for next iteration
#                if self.loss_ == 'deviance':
#                    # prevents overflow at the next step
#                    curr_ttls[curr_ttls[:,k] > 100,k] = 100
#                    p_ = np.exp(curr_ttls[:,k]) / \
#                        (np.exp(curr_ttls[:,k]) + np.exp(-curr_ttls[:,k]))
#                    res_train[:,k] = y_std - p_
#                elif self.loss_ == 'rmse':
#                    res_train[:,k] = robust_sign(y_std - curr_ttls[:,k])
#                i_iter = i_iter + 1
#        # Return
#        if self.loss_ == 'deviance':
#            self.y_pred = np.sign(curr_ttls)
#            self.y_pred[self.y_pred == 0] = -1
#            predictions=np.zeros(X.shape[0],dtype=np.int32)
#            for k in self.ks:
#                predictions += self.y_pred[:,k]>=0 
#            self.y_pred= self.classes_.take(np.asarray(predictions, dtype=np.intp))
#
#        elif self.loss_ == 'rmse':
#            self.y_pred = curr_ttls[:,0]
#
#        return
#
##    def predict_proba(self, X_pred, cum=False):
##        """Predict class or regression value for X.
##        For a classification model, the predicted class for each sample in X is
##        returned. For a regression model, the predicted value based on X is
##        returned.
##        Parameters
##        ----------
##        X : array-like or sparse matrix of shape = [n_samples, n_features]
##            The input samples. Internally, it will be converted to
##            ``dtype=np.float32`` and if a sparse matrix is provided
##            to a sparse ``csr_matrix``.
##        cum : boolean, (default=False)
##            True to include all predictions for all stages.
##        Returns
##        -------
##        y : array of shape = [n_samples] or [n_samples, n_outputs]
##            The predicted classes, or the predict values.
##        """
##        if len(X_pred.shape) == 0:
##            X_pred_ = np.zeros([1, len(X_pred)])
##            X_pred_[0, :] = X_pred
##        else:
##            X_pred_ = X_pred
##        res = np.zeros([X_pred.shape[0], len(self.estimators)]
##                       ) if cum else np.zeros(X_pred.shape[0])
##        num_comp = np.zeros([X_pred.shape[0], len(self.estimators)]
##                            ) if cum else np.zeros(X_pred.shape[0])
##        for i in np.arange(len(self.estimators)):
##            [pred_, is_comp] = self.estimators[i].predict_proba(X_pred)
##            if cum:
##                res[:, i] = res[:, i - 1] + self.eta * pred_
##                num_comp[:, i] = num_comp[:, i - 1] + is_comp
##            else:
##                res = res + self.eta * pred_
##                num_comp = num_comp + is_comp
##        return [res, num_comp]
#    
#    def decision_function(self, X_pred, cum=False):
#        """Predict class or regression value for X.
#        For a classification model, the predicted class for each sample in X is
#        returned. For a regression model, the predicted value based on X is
#        returned.
#        Parameters
#        ----------
#        X : array-like or sparse matrix of shape = [n_samples, n_features]
#            The input samples. Internally, it will be converted to
#            ``dtype=np.float32`` and if a sparse matrix is provided
#            to a sparse ``csr_matrix``.
#        cum : boolean, (default=False)
#            True to include all predictions for all stages.
#        Returns
#        -------
#        y : array of shape = [n_samples] or [n_samples, n_outputs]
#            The predicted classes, or the predict values.
#        """
#        if len(X_pred.shape) == 0:
#            X_pred_ = np.zeros([1, len(X_pred)])
#            X_pred_[0, :] = X_pred
#        else:
#            X_pred_ = X_pred
#        res = {}#np.zeros([X_pred.shape[0], len(self.estimators)]
#                 #      ) if cum else np.zeros(X_pred.shape[0])
#        num_comp = {} #np.zeros([X_pred.shape[0], len(self.estimators)]
#                       #     ) if cum else np.zeros(X_pred.shape[0])
#        for k in self.ks:
#            if cum:
#                res[k]=np.zeros([X_pred.shape[0], len(self.estimators)])
#                num_comp[k]=np.zeros([X_pred.shape[0], len(self.estimators)])
#            else:
#                res[k]=np.zeros(X_pred.shape[0])
#                num_comp[k]=np.zeros(X_pred.shape[0])
#            for i in np.arange(len(self.estimators[k])):
#                mono_learn_binary=self.estimators[k][i]
#                [pred_, is_comp] = mono_learn_binary.decision_function(X_pred)
#                if cum:
#                    res[k][:, i] = res[k][:, i - 1] + self.eta * pred_
#                    num_comp[k][:, i] = num_comp[k][:, i - 1] + is_comp
#                else:
#                    res[k] = res[k] + self.eta * pred_
#                    num_comp[k] = num_comp[k] + is_comp
#            res[k]=res[k]+self.intercept_[k]
#        return [res, num_comp]
#
#    def predict(self, X_pred, cum=False):
#        if len(X_pred.shape)==1:
#            X_pred=X_pred.reshape([1,-1])
#        [dec, comp]=self.decision_function(X_pred, cum)
#        predictions=np.zeros(X_pred.shape[0],dtype=np.int32)
#        if self.loss_=='deviance':
#            for k in self.ks:
#                predictions += dec[k]>=0 
#            
#            return self.classes_.take(np.asarray(predictions, dtype=np.intp))
#        else:
#            predictions=dec[0]
#            return predictions
#        
#
##        maj_class = self.y_maj_class_calc
##        [y, num_comp] = np.sign(self.decision_function(X_pred, cum))
##        if cum:
##            for col in np.arange(y.shape[1]):
##                y[y[:, col] == 0, col] = maj_class
##        else:
##            y[y == 0] = maj_class
##        return y
#
#    def get_all_learners(self):
#        return self.estimators
#
#    def transform(self, X):
#        """Transform dataset.
#
#        Parameters
#        ----------
#        X: array-like matrix
#
#        Returns
#        -------
#        X_transformed: array-like matrix, shape=(n_samples, 1)
#        """
#        return np.array([instance.transform(X)
#                         for instance in self.get_all_learners()]).T

def fit_one_class_svm(delta_X,weights,v,mt_feat_types_):
        
        mt_feat_types=np.asarray(mt_feat_types_,dtype=np.int32)
        N = delta_X.shape[0]
        p = delta_X.shape[1]
        #print(N)
        #num_feats = p
        mt_feats = np.arange(p)[mt_feat_types!=0]#np.asarray(list(incr_feats) + list(decr_feats))
        nmt_feats = np.arange(p)[mt_feat_types==0]#np.asarray(
        #    [j for j in np.arange(num_feats) + 1 if j not in mt_feats])
        solvers.options['show_progress'] = False
        if N == 0:
            return np.zeros(delta_X.shape[1])#[-99]
        else:
            # Build QP matrices
            # Minimize     1/2 x^T P x + q^T x
            # Subject to   G x <= h
            #             A x = b
            if weights is None:
                weights = np.ones(N)
#            P = np.asarray([],dtype=np.float64)
            P = np.zeros([p + 2*N, p + 2*N])
            for ip in nmt_feats :
                P[ip, ip] = 1
            for ip in mt_feats :
                P[ip, ip] = 1
            q=np.zeros(p+2*N)
            q[p:p+N]=v
            q[p+N:]=(1.-v)
            #q = 1 / (N * v) * np.ones((N + p, 1))
            #q[0:p, 0] = 0
            #q[p:, 0] = q[p:, 0] * weights
            G1a = np.zeros([p, p])
            for ip in np.arange(p):
                G1a[ip, ip] = -1 if ip in mt_feats  else 1
            G1 = np.hstack([G1a, np.zeros([p, 2*N])])
            G2 = np.hstack([np.zeros([2*N, p]), -np.eye(2*N)])
            #G2 = np.hstack([np.zeros([N, p]), -np.eye(N),np.zeros([N,N])])
            
            #G2 = np.hstack([np.zeros([N, p]),np.zeros([N,N]), np.eye(N)])
            G3 = np.hstack([-delta_X ,-np.eye(N),np.zeros([N,N])])
            G4 = np.hstack([delta_X ,np.zeros([N,N]),-np.eye(N)])
            G = np.vstack([G1, G2, G3,G4])
            h = np.zeros([p + 4 * N])
            A = np.zeros([1, p + 2*N])
            for ip in np.arange(p):
                A[0, ip] = 1 if ip in mt_feats  else -1
            b = np.asarray([1.])
            #b = np.asarray([0.])
            P = cvxmat(P)
            q = cvxmat(q)
            A = cvxmat(A)
            b = cvxmat(b)
            # options['abstol']=1e-20 #(default: 1e-7).
            # options['reltol']=1e-11 #(default: 1e-6)
            sol = solvers.qp(P, q, cvxmat(G), cvxmat(h), A, b)

            if sol['status'] != 'optimal':
                print(
                    '****** NOT OPTIMAL ' +
                    sol['status'] +
                    ' ******* [N=' +
                    str(N) +
                    ', p=' +
                    str(p) +
                    ']')
                return np.zeros(delta_X.shape[1])-99#[-99]
            else:
                soln = sol['x']
                w = np.ravel(soln[0:p, :])
                if np.sum(np.abs(w))==0.:
                    print('what the?')
                # err = np.asarray(soln[-N:, :])
                return w
            
            
def robust_sign(y):
    y_ = np.sign(y)
    uniq = np.unique(y_)  # .sort()
    if uniq is None:
        print('sdf')
    if 0 in uniq:
        replace = -1 if -1 not in uniq else 1
        y_[y_ == 0] = replace
    return y_
