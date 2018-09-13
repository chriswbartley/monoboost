# cython: cdivision=True, boundscheck=False, wraparound=False, nonecheck=False
# Author: Peter Prettenhofer
#
# License: BSD 3 clause

cimport cython

from libc.stdlib cimport free
from libc.string cimport memset
from libcpp cimport bool
from libc.math cimport exp
from libc.math cimport log
import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

#from sklearn.tree._tree cimport Node
#from sklearn.tree._tree cimport Tree
#from sklearn.tree._tree cimport DTYPE_t
#from sklearn.tree._tree cimport SIZE_t
#from sklearn.tree._tree cimport INT32_t
#from sklearn.tree._utils cimport safe_realloc
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

ctypedef np.int32_t int32
ctypedef np.float64_t float64
ctypedef np.float_t float
ctypedef np.uint8_t uint8

# no namespace lookup for numpy dtype and array creation
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import bool as np_bool
from numpy import float32 as np_float32
from numpy import float64 as np_float64


# constant to mark tree leafs
cdef SIZE_t TREE_LEAF = -1
cdef float64 RULE_LOWER_CONST=-1e9
cdef float64 RULE_UPPER_CONST=1e9

         
@cython.boundscheck(False)
cdef void _apply_rules(float64 *X,
                       float64 *rule_corners_MT,
                           float64 *rule_hyperplanes_NMT,
                           float64 *rule_intercepts_NMT,
                           float64 *mt_feat_types,
                           float64 dirn,
                           int32 strict,
                           Py_ssize_t n_samples,
                          Py_ssize_t n_features,
                          Py_ssize_t n_rules,
                          int32 *out):
    cdef int32 res 
    cdef float64 nmt_dot_sum
    cdef float64 diff
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    for i in range(n_samples):
        for r in range(n_rules):
            res=1
            j=0
            nmt_dot_sum=rule_intercepts_NMT[r]
            while res==1 and j<n_features:
                if mt_feat_types[j]!=0:
                    if strict==0:
                        #if X[i * n_features + j]*dirn*mt_feat_types[j] < rule_corners_MT[r * n_features + j]*dirn:
                        
                        if X[i * n_features + j]*dirn*mt_feat_types[j] < rule_corners_MT[r * n_features + j]*dirn*mt_feat_types[j]:
                            res=0
                    else: # strict
                        if X[i * n_features + j]*dirn*mt_feat_types[j] <= rule_corners_MT[r * n_features + j]*dirn*mt_feat_types[j]:
                            res=0   
                #else: # nmt feat
                diff=X[i * n_features + j]-rule_corners_MT[r * n_features + j]
                if diff<0.:
                    diff=-diff
                nmt_dot_sum=nmt_dot_sum+diff*rule_hyperplanes_NMT[r * n_features + j]
                    
                j=j+1
            if nmt_dot_sum>=0.:
                out[i * n_rules + r]= res
            else:
                out[i * n_rules + r]=0

                          
def apply_rules_c(np.ndarray[float64, ndim=2] X, 
                  object rule_corners_MT, 
                  object rule_hyperplanes_NMT,
                  object rule_intercepts_NMT,
                  object mt_feat_types,
                  float64 dirn,
                  int32 strict,
                   np.ndarray[int32, ndim=2] out):

        _apply_rules(<float64*> (<np.ndarray> X).data, 
                 <float64*> (<np.ndarray> rule_corners_MT).data, 
                 <float64*> (<np.ndarray> rule_hyperplanes_NMT).data,
                 <float64*> (<np.ndarray> rule_intercepts_NMT).data,
                 <float64*> (<np.ndarray> mt_feat_types).data,
                 dirn,
                 strict,
                 X.shape[0],
                 X.shape[1],
                 rule_corners_MT.shape[0],
                 <int32*> (<np.ndarray> out).data)


