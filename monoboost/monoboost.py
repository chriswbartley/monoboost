# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:54:18 2017
 
@author: 19514733
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from cvxopt import matrix as cvxmat, sparse,solvers

__all__ = ["Scale", "MonoComparator", "MonoLearner", "MonoBoost", "MonoBoostEnsemble"]
 
TOL=0 #1e-55
 
class Scale():
    """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5

    Each variable is firsst Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
    Warning: this class should not be used directly.
    """    
    def __init__(self,trim_quantile=0.0):
        self.trim_quantile=trim_quantile
        self.scale_multipliers=None
        self.winsor_lims=None
        
    def train(self,X):
        # get winsor limits
        self.winsor_lims=np.ones([2,X.shape[1]])*np.inf
        self.winsor_lims[0,:]=-np.inf
        if self.trim_quantile>0:
            for i_col in np.arange(X.shape[1]):
                lower=np.percentile(X[:,i_col],self.trim_quantile*100)
                upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
                self.winsor_lims[:,i_col]=[lower,upper]
        # get multipliers
        scale_multipliers=np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals=len(np.unique(X[:,i_col]))
            if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
                X_col_winsorised=X[:,i_col].copy()
                X_col_winsorised[X_col_winsorised<self.winsor_lims[0,i_col]]=self.winsor_lims[0,i_col]
                X_col_winsorised[X_col_winsorised>self.winsor_lims[1,i_col]]=self.winsor_lims[1,i_col]
                scale_multipliers[i_col]=1.0/np.std(X_col_winsorised)
        self.scale_multipliers=scale_multipliers
        
    def scale(self,X):
        return X*self.scale_multipliers
    def unscale(self,X):
        return X/self.scale_multipliers
class MonoComparator():
    def __init__(self,n_feats,incr_feats,decr_feats,nmt_hyperplane=None):
        self.incr_feats=np.asarray(incr_feats)
        self.decr_feats=np.asarray(decr_feats)
        self.nmt_hyperplane=nmt_hyperplane
        self.mt_feats=np.asarray(list(incr_feats)+list(decr_feats))
        self.nmt_feats=np.asarray([j for j in np.arange(n_feats)+1 if j not in self.mt_feats])
        self.n_feats=n_feats
    def compare(self,x1_in,x2_in,check_nmt_feats=True,strict=False):
        # returns: -1 if decreasing, 0 if identical, +1 if increasing, -99 if incomparable
        if len(self.mt_feats)==0:
            return -99
        elif len(x1_in.shape)>1:
            x1=np.ravel(x1_in)
            x2=np.ravel(x2_in)
        else:
            x1=x1_in.copy()
            x2=x2_in.copy()
        # check for identical
        if np.array_equal(x1,x2):
            return 0
        # reverse polarity of decreasing features
        for dec_feat in self.decr_feats:
            x1[dec_feat-1]=-1*x1[dec_feat-1]
            x2[dec_feat-1]=-1*x2[dec_feat-1]
        # check mt feats all increasing (or decreasing)
        mt_feats_difference=np.zeros(self.n_feats)
        if len(self.mt_feats)>0: mt_feats_difference[self.mt_feats-1]=x2[self.mt_feats-1]-x1[self.mt_feats-1]
        mt_feats_same=np.sum(mt_feats_difference[self.mt_feats-1]==0)
        if strict:
            mt_feats_incr=np.sum(mt_feats_difference[self.mt_feats-1]>0) 
            mt_feats_decr=np.sum(mt_feats_difference[self.mt_feats-1]<0)            
        else:
            mt_feats_incr=np.sum(mt_feats_difference[self.mt_feats-1]>=0) 
            mt_feats_decr=np.sum(mt_feats_difference[self.mt_feats-1]<=0) 
        if mt_feats_same==len(self.mt_feats):
            comp=0
        elif mt_feats_incr==len(self.mt_feats): # increasing
            comp=+1       
        elif mt_feats_decr==len(self.mt_feats): # decreasing
            comp=-1
        else: # incomparale
            comp=-99
        # short exit if available
        if comp==-99 or comp==0:
            return -99
        # if still going, check mt feats by weakened planes
        if len(self.nmt_feats)==0 or not check_nmt_feats or self.nmt_hyperplane is None:
            nmt_feat_compliance=True
        else:
            x_diff=np.abs(x2-x1)
            dot_prod=np.dot(self.nmt_hyperplane,x_diff)
            nmt_feat_compliance=dot_prod>=-TOL
        # return result
#        try:
#            if len(nmt_feat_compliance)>0:
#                print('sdf')
#        except:
#            pass
        if nmt_feat_compliance:
            return comp
        else: # incomparable due to nmt features
            return -99
             
class MonoLearner():
    def __init__(self,n_feats,incr_feats,decr_feats,coefs=None, dirn=None,x_base=None,nmt_hyperplane=None, learner_type='two-sided',loss='rmse'):
        self.incr_feats=np.asarray(incr_feats)
        self.decr_feats=np.asarray(decr_feats)
        self.coefs=coefs
        self.dirn=dirn
        self.x_base=x_base
        self.mt_feats=np.asarray(list(incr_feats)+list(decr_feats))
        self.nmt_feats=np.asarray([j for j in np.arange(n_feats)+1 if j not in self.mt_feats])
        self.comparator=MonoComparator(n_feats,incr_feats,decr_feats,nmt_hyperplane)
        self.nmt_hyperplane=nmt_hyperplane
        self.learner_type_code=0 if learner_type=='two-sided' else 1
        self.loss=loss # note loss only affects the calculation of the coefficients - all splits are done RMSE
    @property
    def nmt_hyperplane(self):
        """I'm the 'x' property."""
         
        return self.comparator.nmt_hyperplane #self._nmt_hyperplane
 
    @nmt_hyperplane.setter
    def nmt_hyperplane(self, value):
        self.comparator.nmt_hyperplane=value
        #self._nmt_hyperplane = value
    def predict_proba(self, X_pred):
        if len(X_pred.shape)==1:
            X_pred_=np.zeros([1,len(X_pred)])
            X_pred_[0,:]=X_pred
        else:
            X_pred_=X_pred
             
        y_pred=np.zeros(X_pred_.shape[0])
        is_comp=np.zeros(X_pred_.shape[0])
        for i in np.arange(len(y_pred)):
            comp=self.comparator.compare(self.x_base,X_pred_[i,:])
            is_comp[i]=1 if comp==0 or comp==self.dirn else 0
            y_pred[i]=self.coefs[1] if (comp==0 or comp==self.dirn) else self.coefs[0] 
        #print('comparable: ' + str(num_comp))
        return [y_pred,is_comp]
 
    def fit_from_cache(self,cached_local_hp_data,X,y,res_train,curr_totals,hp_reg=None,hp_reg_c=None):
        best=[1e99,-1,-99,-1,[-1,-1]]    # err, base, dirn, hp, coefs
        for i in np.arange(X.shape[0]):
            data_i=cached_local_hp_data[i]
            for dirn in [-1,+1]:
                data_dirn=data_i[dirn]
                vs=data_dirn['vs']
                hps=data_dirn['hps']
                comp_idxs=data_dirn['comp_idxs']
                for i_v in np.arange(len(vs)):
                    comp_pts=comp_idxs[i_v]
                    incomp_pts=np.asarray(np.setdiff1d(np.arange(X.shape[0]) ,comp_pts))#[j for j in np.setdiff1d(np.arange(X.shape[0]) ,comp_pts)])
                    hp=hps[i_v,:]
                    v=vs[i_v]
                    mean_res_in=np.mean(res_train[comp_pts])
                    mean_res_out=np.mean(res_train[incomp_pts])
                    sse=np.sum((res_train[comp_pts]-mean_res_in)**2)+np.sum((res_train[incomp_pts]-mean_res_out)**2)#np.sqrt(np.sum((res_train[comp_pts]-mean_res_in)**2)+np.sum((res_train[incomp_pts]-mean_res_out)**2))
                    if not hp_reg is None and len(self.nmt_feats)>0:
                        if hp_reg=='L1_nmt' or hp_reg=='L2_nmt':
                            sse=sse+hp_reg_c*np.linalg.norm(hp[self.nmt_feats-1],ord=1 if hp_reg=='L1_nmt' else 2)**(1 if hp_reg=='L1_nmt' else 2)
                        elif hp_reg=='L1' or hp_reg=='L2':
                            sse=sse+hp_reg_c*np.linalg.norm(hp,ord=1 if hp_reg=='L1' else 2)**(1 if hp_reg=='L1' else 2)
                    if sse<=best[0] and len(comp_pts)>0: ## XXX TRY REMOVING len(comp_pts)>0 XXX
                        if self.loss=='deviance':
                            sum_res_comp=np.sum(np.abs(res_train[comp_pts])*(1-np.abs(res_train[comp_pts])))
                            sum_res_incomp=np.sum(np.abs(res_train[incomp_pts])*(1-np.abs(res_train[incomp_pts])))
                            signed_sum_res_comp=np.sum(res_train[comp_pts])
                            signed_sum_res_incomp=np.sum(res_train[incomp_pts])
                            if  (sum_res_comp>1e-9  and sum_res_incomp>1e-9 and np.abs(signed_sum_res_comp)>1e-9 and np.abs(signed_sum_res_incomp)>1e-9) :
                                coef_in=0.5*signed_sum_res_comp/(sum_res_comp)
                                if self.learner_type_code==0: # std two sided
                                    coef_out=0.5*signed_sum_res_incomp/(sum_res_incomp)  
                                    ratio=np.max([np.abs(coef_in/coef_out),np.abs(coef_out/coef_in)])
                                elif self.learner_type_code==1: # one-sided
                                    [coef_out,ratio]=[0.,0.5]
                            else:
                                coef_in=0
                                coef_out=0 
                                ratio=0. 
                        elif self.loss=='rmse':
                            coef_in=np.mean(y[comp_pts]-curr_totals[comp_pts])
                            coef_out=0 if self.learner_type_code==1 else np.median(y[incomp_pts]-curr_totals[incomp_pts])
                            ratio=0. 
                        if np.sign(coef_in)==dirn and coef_in!=np.inf and coef_out!=np.inf and ratio<1e9:
                            best=[sse,i,dirn,hp,[coef_out,coef_in]]    # err, base, dirn, hp
        self.x_base=X[best[1],:]
        self.coefs=best[4]
        self.dirn=best[2]
        self.nmt_hyperplane=best[3]
        #self.comparator.nmt_hyperplane=self.nmt_hyperplane
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
        #rule_applies = [condition.transform(X) for condition in self.conditions]
        #return reduce(lambda x,y: x * y, rule_applies)
        res=np.asarray([1 if self.comparator.compare(self.x_base,X_pred_[i,:])*self.dirn in [0,1] else 0 for i in np.arange(X_pred_.shape[0])])
        
        return res #.reshape([X_pred_.shape[0],1])
         
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
    def __init__(self,n_feats,incr_feats,decr_feats,num_estimators=10,fit_algo='L2-one-class',eta=1.,vs=[0.001,0.1,0.25,0.5,1],verbose=False,hp_reg=None,hp_reg_c=None,incomp_pred_type='default',learner_type='one-sided',random_state=None,standardise=True):
        self.X=None
        self.y=None
        self.n_feats=n_feats
        self.incr_feats=np.asarray(incr_feats)
        self.decr_feats=np.asarray(decr_feats)
        self.mt_feats=np.asarray(list(incr_feats)+list(decr_feats))
        self.nmt_feats=np.asarray([j for j in np.arange(n_feats)+1 if j not in self.mt_feats])
        self.fitted=False
        self.standardise=standardise
        self.fit_algo=fit_algo  
        self.eta=eta
        self.num_estimators=num_estimators
        self.vs=vs
        self.mt_comparator=MonoComparator(n_feats,incr_feats,decr_feats,nmt_hyperplane=None)
        self.verbose=verbose
        self.hp_reg=hp_reg
        self.hp_reg_c=hp_reg_c
        self.y_pred_num_comp_=None
        self.incomp_pred_type=incomp_pred_type
        self.learner_type=learner_type #0 if learner_type=='two-sided' else 1
        self.random_state=np.random.randint(1e6) if random_state is None else random_state
        np.random.seed(self.random_state)
        self.loss='auto'
    @property
    def y_maj_class_calc(self):
        """I'm the 'x' property."""
        return -1 if np.sum(self.y==-1)/len(self.y)>=0.5 else +1   
    @property
    def y_pred_num_comp(self):
        """I'm the 'x' property."""
        if hasattr(self, 'y_pred_num_comp_')==False:
            self.y_pred_num_comp_ = None
        if self.y_pred_num_comp_ is None:
            [ypred,num_comp]=self.predict_proba(self.X)
            self.y_pred_num_comp_=num_comp
        return self.y_pred_num_comp_ 
    def solve_hp(self,incr_feats,decr_feats,delta_X,v,weights=None):
        N=delta_X.shape[0]
        p=delta_X.shape[1]
        num_feats=p
        mt_feats=np.asarray(list(incr_feats)+list(decr_feats))
        nmt_feats=np.asarray([j for j in np.arange(num_feats)+1 if j not in mt_feats])
        solvers.options['show_progress'] = False   
        if N==0:
            return [-99]
        else:
            # Build QP matrices
            #Minimize     1/2 x^T P x + q^T x
            #Subject to   G x <= h  
            #             A x = b
            if weights is None:
                weights=np.ones(N)
            P=np.zeros([p+N,p+N])
            for ip in nmt_feats-1:#np.arange(p):
                P[ip,ip]=1
            q=1/(N*v)*np.ones((N+p,1))
            q[0:p,0]=0
            q[p:,0]=q[p:,0]*weights
            G1a=np.zeros([p,p])
            for ip in np.arange(p):
                G1a[ip,ip]=-1 if ip in mt_feats-1 else 1
            G1=np.hstack([G1a,np.zeros([p,N])])
            G2=np.hstack([np.zeros([N,p]),-np.eye(N)])
            G3=np.hstack([delta_X,-np.eye(N)])
            G=np.vstack([G1,G2,G3])
            h=np.zeros([p+2*N])
            A=np.zeros([1,p+N])
            for ip in np.arange(p):
                A[0,ip]=1 if ip in mt_feats-1 else -1
            b=np.asarray([1.])
            P = cvxmat(P)                                                                                                      
            q = cvxmat(q)
            A = cvxmat(A) # unnecessary
            b = cvxmat(b)
            #options['abstol']=1e-20 #(default: 1e-7).
            #options['reltol']=1e-11 #(default: 1e-6)
            sol = solvers.qp(P, q, cvxmat(G), cvxmat(h), A, b)
            if sol['status']!='optimal':
                print('****** NOT OPTIMAL '+sol['status']+' ******* [N=' + str(N) + ', p=' + str(p)+']'  )
                return [-99]
            else:
                soln=sol['x']
                w=np.ravel(soln[0:p,:])
                err=np.asarray(soln[-N:,:])
                return w 
       
    def get_deltas(self,X_base_pt, X,y):
        dirns=[-1,1]
        deltas=[np.zeros([X.shape[0],X.shape[1]]),np.zeros([X.shape[0],X.shape[1]])]
        comp_indxs=[]
        for dirn in dirns:
            idirn=0 if dirn==-1 else 1
            i_j=0
            comp_indxs_=[]
            for j in np.arange(X.shape[0]):    
                if y[j]==-dirn :
                    comp=self.mt_comparator.compare(X_base_pt,X[j,:])
#                    c_=(X[j,:]-X_base_pt)*np.asarray([1 if ii in self.incr_feats else -1 if ii in self.decr_feats else 0 for ii in np.arange(self.n_feats)+1])
#                    if np.all(c_>=0): 
#                        print('increasing')
#                    elif np.all(c_<=0):
#                        print('dereassing')
                    if comp==dirn or comp==0:
                        comp_indxs_=comp_indxs_+[j]
                        d_=X[j,:]-X_base_pt
                        #if not np.all(d_==0.):
                        deltas[idirn][i_j,:]=np.abs(d_)
                        #deltas[idirn][i_j,self.decr_feats-1]=dirn*deltas[idirn][i_j,self.decr_feats-1]
                        #deltas[idirn][i_j,self.nmt_feats-1]=np.abs(deltas[idirn][i_j,self.nmt_feats-1])
                        i_j=i_j+1
            deltas[idirn]=deltas[idirn][0:i_j,:]
            comp_indxs=comp_indxs+[np.asarray(comp_indxs_)]
        return [comp_indxs,deltas]
    def fit_cache(self,X,y,svm_vs) :
        X_rev_dec=X.copy()
        for dec_feat in self.decr_feats:
            X_rev_dec[:,dec_feat-1]=-1*X_rev_dec[:,dec_feat-1]
        hp_data=dict()
        calc_hyperplanes=self.learner_type!='ignore_nmt_feats'
        for i in np.arange(X.shape[0]):
            dirn_=dict()
            dirn_pos=dict()
            dirn_neg=dict()
            dirn_[-1]=dirn_neg
            dirn_[1]=dirn_pos
            hp_data[i]=dirn_
            x_i=X[i,:]
            # get base comparable pts in given direction
            pts_above=[]
            pts_below=[]
            for j in np.arange(X.shape[0]):
                #if i!=j:
                comp=self.mt_comparator.compare(X[i,:],X[j,:])
                if comp==+1 or comp==0:
                    pts_above=pts_above+[j]
                if comp==-1 or comp==0:
                    pts_below=pts_below+[j]
            hp_data[i][-1]['base_comp_idxs']=np.asarray(pts_below)
            hp_data[i][1]['base_comp_idxs']=np.asarray(pts_above)
            # calculate hyperplanes            
            for dirn in [-1,+1]:
                base_comp_idxs=np.asarray(pts_above if dirn==1 else pts_below)
                hps=np.zeros([0,X.shape[1]])
                if calc_hyperplanes:
                    if len(base_comp_idxs)>0:
                        y_signed=y[base_comp_idxs]
                        if len(y_signed[y_signed*dirn<0])>=2 and len(self.nmt_feats)!=0: # fit svm for each v
                            hps=np.zeros([len(svm_vs),X.shape[1]])
                            comp_idxs=[]
                            vs=[]
                            deltas=np.zeros([len(base_comp_idxs),X.shape[1]])
                            weights=np.zeros(X.shape[0])
                            i_j=0
                            for j in base_comp_idxs:    
                                if np.sign(y[j])==-dirn and j!=i:
                                    deltas[i_j,:]=dirn*(X_rev_dec[j,:]-X_rev_dec[i,:])
                                    weights[i_j]=np.abs(y[j])
                                    for k in self.nmt_feats:
                                        deltas[i_j,k-1]=np.abs(deltas[i_j,k-1])
                                    i_j=i_j+1
                            deltas=deltas[0:i_j,:]
                            weights=weights[0:i_j]
                            i_v_real=0
                            smt_comparator=MonoComparator(self.n_feats,self.incr_feats,self.decr_feats,nmt_hyperplane=None)
                            for i_v in np.arange(len(svm_vs)-1,-1,-1):
                                v=svm_vs[i_v]
                                fitted_hp=self.solve_hp(self.incr_feats,self.decr_feats,deltas,v,weights)
                                if fitted_hp[0]!=-99 and np.sum(np.abs(fitted_hp-hps[i_v_real-1,:]))>5e-4:
                                    smt_comparator.nmt_hyperplane=fitted_hp
                                    comp_pts_v=[]
                                    if len(comp_idxs)==0:
                                        com_inds=base_comp_idxs 
                                    else:
                                        com_inds=comp_idxs[-1]
                                    for c in com_inds:
                                        comp=smt_comparator.compare(x_i,X[c,:])
                                        if comp==dirn or comp==0:
                                            comp_pts_v=comp_pts_v+[c]
                                    hps[i_v_real,:]=fitted_hp
                                    comp_idxs=comp_idxs+[np.asarray(comp_pts_v)]
                                    vs=vs+[v]
                                    i_v_real=i_v_real+1
                            hps=hps[0:i_v_real,:]
                if hps.shape[0]==0: # we have no hyperplanes, add the default (NMT feats are irrelevant)
                    hps=np.zeros([1,X.shape[1]])
                    hps[0,:]=np.asarray([ 1 if kk in self.mt_feats else 0 for kk in np.arange(X.shape[1])+1])/len(self.mt_feats)
                    vs=[-99]
                    comp_idxs=[base_comp_idxs]
  
                hp_data[i][dirn]['hps']=hps 
                hp_data[i][dirn]['vs']=np.asarray(vs )
                hp_data[i][dirn]['comp_idxs']=comp_idxs
        return hp_data  
     
         
    def fit(self, X, y):
        """Fits one hyperplane per non-monotone feature
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
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
        #solvers.options['feastol'] = 1e-16# default: 1e-7 
        # save vars
        self.X=X
        self.y=y
        if self.standardise:
            self.scale=Scale()
            self.scale.train(X)
            self.X_scaled=self.scale.scale(X)
        else:
            self.X_scaled=X
        self.y_pred_num_comp_=None
        self.n_datapts=X.shape[0]
        self.classes=np.sort(np.unique(y))
        self.n_classes=len(self.classes)
        if self.loss=='auto':
            self.loss_='deviance' if self.n_classes<=2 and np.all(self.classes==np.asarray([-1,1])) else  'rmse'
        else:
            self.loss_=self.loss

        #get cache of possible hyperplanes & comparable points
        self.hp_cache=self.fit_cache(self.X_scaled,y, self.vs)
        # start boosting!
        cont=True
        i_iter=0
        y_std=y.copy()
        if self.loss_=='deviance':
            y_std[y_std==-1]=0
            prob_class_one=np.sum(y_std==1)/len(y_std)
            self.intercept_=0.5*(np.log(prob_class_one)-np.log(1-prob_class_one)) #Friedman 2001 equation (29) and Elements of Statistical Learning Algorithm 10.3 Line 1: for binary classification we can do better than initialising to 0 as in Algo 10.4
            res_train=y_std-prob_class_one
        elif self.loss_=='rmse':
            self.intercept_=np.median(y_std)
            res_train=robust_sign(y_std-self.intercept_)

        curr_ttls=np.zeros(X.shape[0])+self.intercept_    
        self.estimators=[]
        self.train_err_all=np.zeros(self.num_estimators)
        self.y_pred_train_all=np.zeros([X.shape[0],self.num_estimators])
        
        while cont:
            # find next best rule 
            est=MonoLearner(self.n_feats,self.incr_feats,self.decr_feats,learner_type=self.learner_type,loss=self.loss_)
            est.fit_from_cache(self.hp_cache,self.X_scaled,y_std,res_train,curr_ttls,self.hp_reg,self.hp_reg_c)
            if    est.dirn==-99 :
                if len(self.estimators)<self.num_estimators:
                    self.num_estimators=len(self.estimators)
                    print('Only made it to ' + str(len(self.estimators)) + ' iterations and couldnt find more viable splits')
                cont=False
            

            else:
                #print([est.coefs,est.dirn,est.x_base,est.nmt_hyperplane])
                [pred_,is_comp]=est.predict_proba(self.X_scaled)
                curr_ttls=curr_ttls+self.eta*pred_
                if self.standardise: # unstandardise the estimator so it can be used and interpreted raw
                    est.x_base=self.scale.unscale(est.x_base)
                    est.nmt_hyperplane=self.scale.scale(est.nmt_hyperplane)
                self.estimators=self.estimators+[est]
                
                # calc next iter
                if self.loss_=='deviance':
                    curr_ttls[curr_ttls>100]=100 # prevents overflow at the next step
                    p_=np.exp(curr_ttls)/(np.exp(curr_ttls)+np.exp(-curr_ttls))
                    res_train=y_std-p_
                    self.y_pred_train_all[:,i_iter]=np.sign(curr_ttls+1e-9)
                    self.train_err_all[i_iter]=np.sum(self.y_pred_train_all[:,i_iter]!=y)/len(y)
                    if self.verbose: print(np.sum(self.y_pred_train_all[:,i_iter]!=y)/len(y))
                elif self.loss_=='rmse':
                    res_train=robust_sign(y_std-curr_ttls)
                cont = i_iter<(self.num_estimators-1)
                       
#                y_pred_test=predict_ensemble(snmi,X_test,ensemble,eta)
#                train_err=train_err+[np.sum(y_pred_train!=y_train)/len(y_train)]
#                test_err=test_err+[np.sum(y_pred_test!=y_test)/len(y_test)]
                i_iter=i_iter+1
        if self.loss_=='deviance':
            self.y_pred=np.sign(curr_ttls)
            self.y_pred[self.y_pred==0]=-1
        elif self.loss_=='rmse':
            self.y_pred=curr_ttls
        return
 
    def predict_proba(self, X_pred,cum=False):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        loo : boolean, (default=False)
            True to exclude one matchng datapoint from training set when doing prediction.
            Effectively Leave-One-Out cross validation.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        if len(X_pred.shape)==0:
            X_pred_=np.zeros([1,len(X_pred)])
            X_pred_[0,:]=X_pred
        else:
            X_pred_=X_pred
        res=np.zeros([X_pred.shape[0],len(self.estimators)]) if cum else np.zeros(X_pred.shape[0]) 
        res=res+self.intercept_
        num_comp=np.zeros([X_pred.shape[0],len(self.estimators)]) if cum else np.zeros(X_pred.shape[0]) 
        for i in np.arange(len(self.estimators)):   
            [pred_,is_comp]=self.estimators[i].predict_proba(X_pred)             
            if cum:
                res[:,i]=res[:,i-1]+self.eta*pred_
                num_comp[:,i]=num_comp[:,i-1]+is_comp
            else:
                res=res+self.eta*pred_
                num_comp=num_comp+is_comp
             
         
        return [res,num_comp]
    def predict(self, X_pred,cum=False):
        [pred,comp]=self.predict_proba(X_pred,cum)
        return np.sign(pred) if self.loss_=='deviance' else pred
        
    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        return np.array([instance.transform(X) for instance in self.estimators]).T

#        self.tmp_num_comparable_estimators=num_comp.copy()
#        self.incomp_pred_type='none'#'local_partial_order' none
#        num_pred_special=0
#        if self.incomp_pred_type=='local_partial_order':
#            [ypred_orig,num_comp_orig]=self.predict_proba(self.X)
#            indx_train_comp=np.arange(self.X.shape[0])[self.y_pred_num_comp>0]
#            y_pred_comp=self.y_pred[indx_train_comp]
#            X_comp=self.X[indx_train_comp,:]
#            kNN=KNeighborsClassifier(n_neighbors=5)
#            kNN.fit(X_comp,y_pred_comp)
#            y_pred_kNN= kNN.predict(X_pred)
#            for j in np.arange(len(y)):
#                if num_comp[j]==0:
#                    # fit local hp in each direction, and find points within each cone
#                    [comp_indxs,deltas]=self.get_deltas(X_pred[j,:], self.X,self.y)
#                    defining_classes=[0,0]
#                    for idirn in [0,1]:
#                        if len(comp_indxs[idirn])>0:
#                            hp=self.solve_hp(self.incr_feats,self.decr_feats,deltas[idirn],self.vs[0])
#                            if hp[0]==-99:
#                                defining_classes[idirn]=99
#                            else:
#                                y_def=-1 if idirn==0 else 1 # provides no information
#                                for k in comp_indxs[idirn]:
#                                    if num_comp_orig[k]>0:#self.pred_y_num_comp[k]>0:
#                                        x_diff=np.abs(self.X[k,:]-X_pred[j,:])
#                                        dot_prod=np.dot(hp,x_diff)
#                                        nmt_feat_compliance=dot_prod>=-TOL
#                                        if nmt_feat_compliance >=0:
#                                            if self.y[k]!=y_def:
#                                                defining_classes[idirn]=self.y_pred[k]
#                                                break # no need to go any further
#                                    
#                    # resolve lower and upper regions
#                    if defining_classes[0]==+1:
#                        y[j]=1
#                        num_pred_special=num_pred_special+1
#                    elif defining_classes[1]==-1:
#                        y[j]=-1
#                        num_pred_special=num_pred_special+1
#                    else:
#                        y[j]=y_pred_kNN[j]#self.y_maj_class
#            print('prop special: ' + str(num_pred_special) + '/' + str(np.sum(num_comp==0)))
#        return y
 
class MonoBoostEnsemble():
    """ Partially Monotone Boosting classifier ensemble
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
    def __init__(self,n_feats,incr_feats,decr_feats,num_estimators=10,fit_algo='L2-one-class',eta=1.,vs=[0.001,0.1,0.25,0.5,1],verbose=False,learner_incomp_pred_type='default',learner_type='one-sided',learner_num_estimators=20,learner_eta=1.0, learner_v_mode='random',sample_fract=1.0,random_state=None,standardise=True):
        self.X=None
        self.y=None
        self.n_feats=n_feats
        self.incr_feats=np.asarray(incr_feats)
        self.decr_feats=np.asarray(decr_feats)
        self.mt_feats=np.asarray(list(incr_feats)+list(decr_feats))
        self.nmt_feats=np.asarray([j for j in np.arange(n_feats)+1 if j not in self.mt_feats])
        self.fitted=False
        self.fit_algo=fit_algo  
        self.eta=eta
        self.num_estimators=num_estimators
        self.vs=vs
        self.verbose=verbose
        self.y_pred_num_comp_=None
        self.learner_incomp_pred_type=learner_incomp_pred_type
        self.learner_type=learner_type #0 if learner_type=='two-sided' else 1
        self.learner_num_estimators=learner_num_estimators
        self.learner_eta=learner_eta
        self.learner_v_mode=learner_v_mode
        self.sample_fract=sample_fract  
        self.random_state=np.random.int(1e6) if random_state is None else random_state
        self.standardise=standardise
        np.random.seed(self.random_state)
        self.loss='auto'
    def fit(self, X, y):
        """Fits one hyperplane per non-monotone feature
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
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
        # save vars
        self.X=X
        self.y=y
        self.y_pred_num_comp_=None
        self.n_datapts=X.shape[0]
        self.classes=np.sort(np.unique(y))
        self.n_classes=len(self.classes)
        if self.loss=='auto':
            self.loss_='deviance' if self.n_classes<=2 and np.all(self.classes==np.asarray([-1,1])) else  'rmse'
        else:
            self.loss_=self.loss
        
        # start boosting!
        i_iter=0
        y_std=y.copy()
        if self.loss_=='deviance':
            y_std[y_std==-1]=0
            prob_class_one=np.sum(y_std==1)/len(y_std)
            self.intercept_=0.5*(np.log(prob_class_one)-np.log(1-prob_class_one)) #Friedman 2001 equation (29) and Elements of Statistical Learning Algorithm 10.3 Line 1: for binary classification we can do better than initialising to 0 as in Algo 10.4
            res_train=y_std-prob_class_one
        elif self.loss_=='rmse':
            self.intercept_=np.median(y_std)
            res_train=robust_sign(y_std-self.intercept_)
        self.estimators=[]
        curr_ttls=np.zeros(X.shape[0])+self.intercept_
        if self.standardise:
            self.scale=Scale()
            self.scale.train(X)
            X_scaled=self.scale.scale(X)
        else:
            X_scaled=X
        while len(self.estimators)<=self.num_estimators:
            # find next best rule 
            if self.learner_v_mode=='random':
                vs_this=self.vs[np.random.randint(len(self.vs))]
            else:
                vs_this=self.vs
            est=MonoBoost(n_feats=self.n_feats,incr_feats=self.incr_feats,decr_feats=self.decr_feats,num_estimators=self.learner_num_estimators,fit_algo=self.fit_algo,eta=self.learner_eta,vs=[vs_this],verbose=self.verbose,hp_reg=None,hp_reg_c=None,incomp_pred_type=self.learner_incomp_pred_type,learner_type=self.learner_type,random_state=self.random_state+np.random.randint(1e4),standardise=False)
            if  self.sample_fract<1:
                sample_indx=np.random.permutation(np.arange(X.shape[0]))[0:int(np.floor(self.sample_fract*X.shape[0]))]
            else:
                sample_indx=np.arange(X.shape[0])
            X_sub=X_scaled[sample_indx,:]
            res_train_sub=res_train[sample_indx]
            est.fit(X_sub,res_train_sub)
            ## Extract estimators - recalculate coefficients based on deviance loss
            for est_monolearn in est.estimators:
                comp_pts_indx=np.arange(X_sub.shape[0])[np.asarray([True if est_monolearn.comparator.compare(est_monolearn.x_base,X_sub[i_,:])*est_monolearn.dirn in [0,1] else False for i_ in np.arange(X_sub.shape[0])])]
                if self.loss_=='deviance':
                    coef_in=0.5*np.sum(res_train_sub[comp_pts_indx])/np.sum(np.abs(res_train_sub[comp_pts_indx])*(1-np.abs(res_train_sub[comp_pts_indx])))
                elif self.loss_=='rmse':
                    coef_in=np.median(y_std[comp_pts_indx]-curr_ttls[comp_pts_indx])
                est_monolearn.coefs=[0,coef_in]
                
                ## Update function totals (for ALL, not just subsample)
                [pred_,is_comp]=est_monolearn.predict_proba(X_scaled)
                curr_ttls=curr_ttls+self.eta*pred_
                if self.standardise: # unscale this estimator so it can be used raw
                    est_monolearn.x_base=self.scale.unscale(est_monolearn.x_base)
                    est_monolearn.nmt_hyperplane=self.scale.scale(est_monolearn.nmt_hyperplane)
                self.estimators=self.estimators+[est_monolearn]
                #[pred_,is_comp]=est_monolearn.predict_proba(X_sub[comp_pts_indx])
                #curr_ttls[sample_indx[comp_pts_indx]]=curr_ttls[sample_indx][comp_pts_indx]+self.eta*pred_
            ## Update res_train for next iteration
            if self.loss_=='deviance':
                curr_ttls[curr_ttls>100]=100 # prevents overflow at the next step
                p_=np.exp(curr_ttls)/(np.exp(curr_ttls)+np.exp(-curr_ttls))
                res_train=y_std-p_
            elif self.loss_=='rmse':
                res_train=robust_sign(y_std-curr_ttls)
            i_iter=i_iter+1
        ## Return
        if self.loss_=='deviance':
            self.y_pred=np.sign(curr_ttls)
            self.y_pred[self.y_pred==0]=-1
        elif self.loss_=='rmse':
            self.y_pred=curr_ttls
         
        return 

    def predict_proba(self, X_pred,cum=False):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        loo : boolean, (default=False)
            True to exclude one matchng datapoint from training set when doing prediction.
            Effectively Leave-One-Out cross validation.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        if len(X_pred.shape)==0:
            X_pred_=np.zeros([1,len(X_pred)])
            X_pred_[0,:]=X_pred
        else:
            X_pred_=X_pred
        res=np.zeros([X_pred.shape[0],len(self.estimators)]) if cum else np.zeros(X_pred.shape[0]) 
        num_comp=np.zeros([X_pred.shape[0],len(self.estimators)]) if cum else np.zeros(X_pred.shape[0]) 
        for i in np.arange(len(self.estimators)):   
            [pred_,is_comp]=self.estimators[i].predict_proba(X_pred)             
            if cum:
                res[:,i]=res[:,i-1]+self.eta*pred_
                num_comp[:,i]=num_comp[:,i-1]+is_comp
            else:
                res=res+self.eta*pred_
                num_comp=num_comp+is_comp
        return [res,num_comp]
        
    def predict(self, X_pred,cum=False):
        maj_class=self.y_maj_class_calc
        [y,num_comp]=np.sign(self.predict_proba(X_pred,cum))
        if cum:
            for col in np.arange(y.shape[1]):
                y[y[:,col]==0,col]=maj_class
        else:
            y[y==0]=maj_class
        return y    
    def get_all_learners(self):
        return self.estimators
    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        return np.array([instance.transform(X) for instance in self.get_all_learners()]).T

def robust_sign(y):
    y_=np.sign(y)
    uniq=np.unique(y_)#.sort()
    if uniq is None:
        print('sdf')
    if 0 in uniq:
        replace=-1 if -1 not in uniq else 1
        y_[y_==0]=replace
    return y_
def _convert(H, f, A, b, Aeq, beq, lb, ub):                                                                                  
    """
    Convert everything to                                                                                              
    cvxopt-style matrices                                                                                              
    """
    P = cvxmat(H)                                                                                                      
    q = cvxmat(f)
    if Aeq is None:                                                                                                    
        A_ = None                                                                                                      
    else: 
        A_ = cvxmat(Aeq)                                                                                                
    if beq is None:                                                                                                    
        b_ = None                                                                                                      
    else: 
        b_ = cvxmat(beq)                                                                                                
     
    if lb is None and ub is None:
        if A is None:
            G=None
            h=None
        else:
            G = cvxmat(A)
            h = cvxmat(b)
    else:
        n=len(lb)
        if A is None:
            G = sparse([-speye(n), speye(n)])     
            h = cvxmat(np.vstack([-lb, ub])) 
        else:
            G = sparse([cvxmat(A), -speye(n), speye(n)]) 
            h = cvxmat(np.vstack([b,-lb, ub])) 
 
    return P, q, G, h, A_, b_
#from __future__ import absolute_import, division, print_function
#import numpy as np
#import pandas as pd
#import scipy.optimize as opt
#from scipy.special import erf
#from .due import due, Doi
#
#__all__ = ["Model", "Fit", "opt_err_func", "transform_data", "cumgauss"]
#
#
## Use duecredit (duecredit.org) to provide a citation to relevant work to
## be cited. This does nothing, unless the user has duecredit installed,
## And calls this with duecredit (as in `python -m duecredit script.py`):
#due.cite(Doi("10.1167/13.9.30"),
#         description="Template project for small scientific Python projects",
#         tags=["reference-implementation"],
#         path='monoboost')
#
#
#def transform_data(data):
#    """
#    Function that takes experimental data and gives us the
#    dependent/independent variables for analysis.
#
#    Parameters
#    ----------
#    data : Pandas DataFrame or string.
#        If this is a DataFrame, it should have the columns `contrast1` and
#        `answer` from which the dependent and independent variables will be
#        extracted. If this is a string, it should be the full path to a csv
#        file that contains data that can be read into a DataFrame with this
#        specification.
#
#    Returns
#    -------
#    x : array
#        The unique contrast differences.
#    y : array
#        The proportion of '2' answers in each contrast difference
#    n : array
#        The number of trials in each x,y condition
#    """
#    if isinstance(data, str):
#        data = pd.read_csv(data)
#
#    contrast1 = data['contrast1']
#    answers = data['answer']
#
#    x = np.unique(contrast1)
#    y = []
#    n = []
#
#    for c in x:
#        idx = np.where(contrast1 == c)
#        n.append(float(len(idx[0])))
#        answer1 = len(np.where(answers[idx[0]] == 1)[0])
#        y.append(answer1 / n[-1])
#    return x, y, n
#
#
#def cumgauss(x, mu, sigma):
#    """
#    The cumulative Gaussian at x, for the distribution with mean mu and
#    standard deviation sigma.
#
#    Parameters
#    ----------
#    x : float or array
#       The values of x over which to evaluate the cumulative Gaussian function
#
#    mu : float
#       The mean parameter. Determines the x value at which the y value is 0.5
#
#    sigma : float
#       The variance parameter. Determines the slope of the curve at the point
#       of Deflection
#
#    Returns
#    -------
#
#    g : float or array
#        The cumulative gaussian with mean $\\mu$ and variance $\\sigma$
#        evaluated at all points in `x`.
#
#    Notes
#    -----
#    Based on:
#    http://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
#
#    The cumulative Gaussian function is defined as:
#
#    .. math::
#
#        \\Phi(x) = \\frac{1}{2} [1 + erf(\\frac{x}{\\sqrt{2}})]
#
#    Where, $erf$, the error function is defined as:
#
#    .. math::
#
#        erf(x) = \\frac{1}{\\sqrt{\\pi}} \int_{-x}^{x} e^{t^2} dt
#
#    """
#    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))
#
#
#def opt_err_func(params, x, y, func):
#    """
#    Error function for fitting a function using non-linear optimization.
#
#    Parameters
#    ----------
#    params : tuple
#        A tuple with the parameters of `func` according to their order of
#        input
#
#    x : float array
#        An independent variable.
#
#    y : float array
#        The dependent variable.
#
#    func : function
#        A function with inputs: `(x, *params)`
#
#    Returns
#    -------
#    float array
#        The marginals of the fit to x/y given the params
#    """
#    return y - func(x, *params)
#
#
#class Model(object):
#    """Class for fitting cumulative Gaussian functions to data"""
#    def __init__(self, func=cumgauss):
#        """ Initialize a model object.
#
#        Parameters
#        ----------
#        data : Pandas DataFrame
#            Data from a subjective contrast judgement experiment
#
#        func : callable, optional
#            A function that relates x and y through a set of parameters.
#            Default: :func:`cumgauss`
#        """
#        self.func = func
#
#    def fit(self, x, y, initial=[0.5, 1]):
#        """
#        Fit a Model to data.
#
#        Parameters
#        ----------
#        x : float or array
#           The independent variable: contrast values presented in the
#           experiment
#        y : float or array
#           The dependent variable
#
#        Returns
#        -------
#        fit : :class:`Fit` instance
#            A :class:`Fit` object that contains the parameters of the model.
#
#        """
#        params, _ = opt.leastsq(opt_err_func, initial,
#                                args=(x, y, self.func))
#        return Fit(self, params)
#
#
#class Fit(object):
#    """
#    Class for representing a fit of a model to data
#    """
#    def __init__(self, model, params):
#        """
#        Initialize a :class:`Fit` object.
#
#        Parameters
#        ----------
#        model : a :class:`Model` instance
#            An object representing the model used
#
#        params : array or list
#            The parameters of the model evaluated for the data
#
#        """
#        self.model = model
#        self.params = params
#
#    def predict(self, x):
#        """
#        Predict values of the dependent variable based on values of the
#        indpendent variable.
#
#        Parameters
#        ----------
#        x : float or array
#            Values of the independent variable. Can be values presented in
#            the experiment. For out-of-sample prediction (e.g. in
#            cross-validation), these can be values
#            that were not presented in the experiment.
#
#        Returns
#        -------
#        y : float or array
#            Predicted values of the dependent variable, corresponding to
#            values of the independent variable.
#        """
#        return self.model.func(x, *self.params)
