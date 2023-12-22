#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from scipy import linalg
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA
from utils import iterative_A
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.naive_bayes import GaussianNB,CategoricalNB,ComplementNB
import torch
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import time
import scipy 

def to_torch(ndarray):
    from collections.abc import Sequence
    if ndarray is None: return None
    if isinstance(ndarray, Sequence):
        return [to_torch(ndarray_) for ndarray_ in ndarray if ndarray_ is not None]
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    if torch.is_tensor(ndarray): return ndarray
    raise ValueError('fail convert')

def _cov(X, shrinkage=-1):
    emp_cov = np.cov(np.asarray(X).T, bias=1)
    if shrinkage < 0:
        return emp_cov
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov


def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    means ： array-like of shape (n_classes, n_features)
        Outer classes means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]

    means_ = np.zeros(shape=(len(classes), X.shape[1]))
    for i in range(len(classes)):
        means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)    
    return means, means_


def split_data(data: np.ndarray, percent_train: float):
    split = data.shape[0] - int(percent_train * data.shape[0])
    return data[:split], data[split:]


def feature_reduce(features: np.ndarray, f: int=None):
    """
        Use PCA to reduce the dimensionality of the features.
        If f is none, return the original features.
        If f < features.shape[0], default f to be the shape.
	"""
    if f is None:
        return features
    if f > features.shape[0]:
        f = features.shape[0]
    
    return sklearn.decomposition.PCA(
        n_components=f,
        svd_solver='randomized',
        random_state=1919,
        iterated_power=1).fit_transform(features)


class TransferabilityMethod:	
    def __call__(self, 
        features: np.ndarray, y: np.ndarray,
                ) -> float:
        self.features = features		
        self.y = y
        return self.forward()

    def forward(self) -> float:
        raise NotImplementedError


class PARC(TransferabilityMethod):
	
    def __init__(self, n_dims: int=None, fmt: str=''):
        self.n_dims = n_dims
        self.fmt = fmt

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)
        
        num_classes = len(np.unique(self.y, return_inverse=True)[0])
        labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y

        return self.get_parc_correlation(self.features, labels)

    def get_parc_correlation(self, feats1, labels2):
        scaler = sklearn.preprocessing.StandardScaler()

        feats1  = scaler.fit_transform(feats1)

        rdm1 = 1 - np.corrcoef(feats1)
        rdm2 = 1 - np.corrcoef(labels2)
        
        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)
        
        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100

    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]

from sklearn.metrics.pairwise import pairwise_kernels
class kernel_FDA:

    def __init__(self, n_components=None, kernel=None):
        self.n_components = n_components
        self.Theta = None
        self.X_train = None
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = 'linear'

    def fit_transform(self, X, y):
        # X: columns are sample, rows are features
        self.fit(X=X, y=y)
        X_transformed = self.transform(X=X, y=y)
        return X_transformed

    def fit(self, X, y):

        score=0
        iter=0
        
        self.clf = GaussianNB()
        for ii in range(0,len(X),500):
          
          print(ii)
          X_batch=X[ii:ii+500].T
          y_batch=y[ii:ii+500]
          self.X_train = X_batch


          # ------ Separate classes:
          X_separated_classes = self._separate_samples_of_classes(X=X_batch, y=y_batch)
          y_batch = np.asarray(y_batch)
          y_batch = y_batch.reshape((1, -1))
          n_samples = X_batch.shape[1]
          labels_of_classes = list(set(y_batch.ravel()))
          n_classes = len(labels_of_classes)
          # ------ M_*:
          Kernel_allSamples_allSamples = pairwise_kernels(X=X_batch.T, Y=X_batch.T, metric=self.kernel)
          M_star = Kernel_allSamples_allSamples.sum(axis=1)
          M_star = M_star.reshape((-1,1))
          M_star = (1 / n_samples) * M_star
          # ------ M_c and M:
          M = np.zeros((n_samples, n_samples))
          for class_index in range(n_classes):
              X_class = X_separated_classes[class_index]
              n_samples_of_class = X_class.shape[1]
              # ------ M_c:
              Kernel_allSamples_classSamples = pairwise_kernels(X=X_batch.T, Y=X_class.T, metric=self.kernel)
              M_c = Kernel_allSamples_classSamples.sum(axis=1)
              M_c = M_c.reshape((-1, 1))
              M_c = (1 / n_samples_of_class) * M_c
              # ------ M:
              M = M + n_samples_of_class * (M_c - M_star).dot((M_c - M_star).T)
          # ------ N:
          N = np.zeros((n_samples, n_samples))
          for class_index in range(n_classes):
              X_class = X_separated_classes[class_index]
              n_samples_of_class = X_class.shape[1]
              Kernel_allSamples_classSamples = pairwise_kernels(X=X_batch.T, Y=X_class.T, metric=self.kernel)
              K_c = Kernel_allSamples_classSamples
              H_c = np.eye(n_samples_of_class) - (1 / n_samples_of_class) * np.ones((n_samples_of_class, n_samples_of_class))
              N = N + K_c.dot(H_c).dot(K_c.T)
          
          # ------ kernel Fisher directions:
          epsilon = 0.00001  #--> to prevent singularity of matrix N
          eig_val, eig_vec = LA.eigh(inv(N + epsilon*np.eye(N.shape[0])).dot(M))
          idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
          eig_val = eig_val[idx]
          eig_vec = eig_vec[:, idx]
          # if self.n_components is not None:
          #     Theta = eig_vec[:, :self.n_components]
          # else:
          #     Theta = eig_vec[:, :n_classes-1]
          
          self.Theta = eig_vec
          Xnew=self.transform(X_batch, y_batch.ravel())
          


          self.clf.fit(Xnew.T, y_batch.ravel())
          score+=self.clf.score(Xnew.T,y_batch.ravel())
          iter+=1
          
          # self.scalings_ = self.Theta
          # print('!!!!!!!')
          # # print(np.asarray(M_star),np.asarray(eig_vec).shape())
          # self.coef_ = np.dot(M_star.T, eig_vec).dot(eig_vec.T)
          
          # self.intercept_ = -0.5 * np.diag(np.dot(M_star.T, self.coef_.T)) + np.log(
          #     self.priors_
          # )
        score/=iter
        return score

    def transform(self, X, y):
        # X: columns are sample, rows are features
        # X_transformed: columns are sample, rows are features
        Kernel_train_input = pairwise_kernels(X=self.X_train.T, Y=X.T, metric=self.kernel)
        X_transformed = (self.Theta.T).dot(Kernel_train_input)
        return X_transformed

    def transform_outOfSample_all_together(self, X, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        Kernel_train_input = pairwise_kernels(X=self.X_train.T, Y=X.T, metric=self.kernel)
        X_transformed = (self.Theta.T).dot(Kernel_train_input)
        return X_transformed

    def _build_kernel_matrix(self, X, kernel_func, option_kernel_func=None):  # --> K = self._build_kernel_matrix(X=X, kernel_func=self._radial_basis)
        # https://stats.stackexchange.com/questions/243104/how-to-build-and-use-the-kernel-trick-manually-in-python
        # X = X.T
        n_samples = X.shape[1]
        n_features = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            xi = X[:, i]
            for j in range(n_samples):
                xj = X[:, j]
                K[i, j] = kernel_func(xi, xj, option_kernel_func)
        return K

    def _radial_basis(self, xi, xj, gamma=None):
        if gamma is None:
            n_features = xi.shape[0]
            gamma = 1 / n_features
        r = (np.exp(-gamma * (LA.norm(xi - xj) ** 2)))
        return r

    def _separate_samples_of_classes(self, X, y):
        # X --> rows: features, columns: samples
        # X_separated_classes --> rows: features, columns: samples
        X = X.T
        y = np.asarray(y)
        y = y.reshape((-1, 1))
        yX = np.column_stack((y, X))
        yX = yX[yX[:, 0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
        y = yX[:, 0]
        X = yX[:, 1:]
        labels_of_classes = list(set(y))
        number_of_classes = len(labels_of_classes)
        dimension_of_data = X.shape[1]
        X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        class_index = 0
        index_start_new_class = 0
        n_samples = X.shape[0]
        for sample_index in range(1, n_samples):
            if y[sample_index] != y[sample_index - 1] or sample_index == n_samples-1:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
                index_start_new_class = sample_index
                class_index = class_index + 1
        for class_index in range(number_of_classes):
            X_class = X_separated_classes[class_index]
            X_separated_classes[class_index] = X_class.T
        return X_separated_classes

    def predict_proba(self, X,y):
        # X_ = pairwise_kernels(X, metric= 'linear')
        X=X.T
        Kernel_allSamples_allSamples = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        # scores = np.dot(Kernel_allSamples_allSamples, self.coef_.T) + self.intercept_
        # scores = self.softmax(scores)
        # scores = self.cls.score(pd.DataFrame(X), pd.DataFrame(y))
        # print(scores,scores_reg)
        return scores ,scores

    def softmax(slf,X, copy=True):
      if copy:
          X = np.copy(X)
      max_prob = np.max(X, axis=1).reshape((-1, 1))
      X -= max_prob
      np.exp(X, X)
      sum_prob = np.sum(X, axis=1).reshape((-1, 1))
      X /= sum_prob
      return X


class My_FDA:

    def __init__(self, n_components=None, kernel=None):
        self.n_components = n_components
        self.U = None
        self.X_train = None
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = 'linear'

    def fit_transform(self, X, y):
        # X: columns are sample, rows are features
        self.fit(X=X, y=y)
        X_transformed = self.transform(X=X, y=y)
        return X_transformed
    def energy_score(self, logits):
        logits = to_torch(logits)
        return torch.logsumexp(logits, dim=-1).numpy()

    def fit(self, X, y):
        
        # self.clf = GaussianNB()
        # self.clf.fit(X, y)
        # logits=self.clf.predict_logits(X)
        # score_e=self.energy_score(logits)
        # idx=score_e.argsort()
        # idx=np.concatenate((np.arange(len(idx)//3),np.arange(2*len(idx)//3,len(idx))))
        # X=X[idx]
        # y=y[idx]
            
        
        # X: columns are sample, rows are features
        X=X.T
        self.X_train = X
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n_samples = X.shape[1]
        n_dimensions = X.shape[0]
        labels_of_classes = list(set(y.ravel()))
        n_classes = len(labels_of_classes)
        # ------ S_B:
        mean_of_total = X.mean(axis=1)
        mean_of_total = mean_of_total.reshape((-1, 1))
        S_B = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            mean_of_class = X_class.mean(axis=1)
            mean_of_class = mean_of_class.reshape((-1, 1))
            temp = mean_of_class - mean_of_total
            S_B = S_B + (n_samples_of_class * temp.dot(temp.T))
        # ------ M_c and M:
        S_W = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            mean_of_class = X_class.mean(axis=1)
            mean_of_class = mean_of_class.reshape((-1, 1))
            for sample_index in range(n_samples_of_class):
                sample_of_class = X_class[:, sample_index]
                sample_of_class = sample_of_class.reshape((-1, 1))
                temp = sample_of_class - mean_of_class
                S_W = S_W + temp.dot(temp.T)
        # ------ Fisher directions:
        epsilon = 0.00001  #--> to prevent singularity of matrix N
        eig_val, eig_vec = LA.eigh(inv(S_W + epsilon*np.eye(S_W.shape[0])).dot(S_B))
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            U = eig_vec[:, :self.n_components]
        else:
            U = eig_vec[:, :n_classes-1]
        self.U = U
        Xnew=self.transform(X, y.ravel())
        
        # from sklearn.naive_bayes import GaussianNB
        self.clf = GaussianNB()
        self.clf.fit(Xnew.T, y.ravel())
        score=self.clf.score(Xnew.T,y.ravel())
        
        
        # self.scalings_ = self.Theta
        # print('!!!!!!!')
        # # print(np.asarray(M_star),np.asarray(eig_vec).shape())
        # self.coef_ = np.dot(M_star.T, eig_vec).dot(eig_vec.T)
        
        # self.intercept_ = -0.5 * np.diag(np.dot(M_star.T, self.coef_.T)) + np.log(
        #     self.priors_
        # )
        return score

    def transform(self, X, y):
        # X: columns are sample, rows are features
        # X_transformed: columns are sample, rows are features
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        return X_reconstructed

    def transform_outOfSample_all_together(self, X, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def _separate_samples_of_classes(self, X, y):
        # X --> rows: features, columns: samples
        # X_separated_classes --> rows: features, columns: samples
        X = X.T
        y = np.asarray(y)
        y = y.reshape((-1, 1))
        yX = np.column_stack((y, X))
        yX = yX[yX[:, 0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
        y = yX[:, 0]
        X = yX[:, 1:]
        labels_of_classes = list(set(y))
        number_of_classes = len(labels_of_classes)
        dimension_of_data = X.shape[1]
        X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        class_index = 0
        index_start_new_class = 0
        n_samples = X.shape[0]
        for sample_index in range(1, n_samples):
            if y[sample_index] != y[sample_index - 1] or sample_index == n_samples-1:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
                index_start_new_class = sample_index
                class_index = class_index + 1
        for class_index in range(number_of_classes):
            X_class = X_separated_classes[class_index]
            X_separated_classes[class_index] = X_class.T
        return X_separated_classes


class SFDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        
    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(_cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = _cov(X, shrinkage=self.shrinkage) 

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter

        # evals, evecs = linalg.eigh(Sb)
        evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw)@Sb)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N

        '''
        self.classes_ = np.unique(y)
        #n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage,)

        return self
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        # return X_new[:, : self._max_components]
        return X_new

    def predict_proba(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return softmax(scores)
        # return score


class LDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components

    def _cov(self,X, shrinkage=-1):
      emp_cov = np.cov(np.asarray(X).T, bias=1)
      if shrinkage < 0:
          return emp_cov
      n_features = emp_cov.shape[0]
      mu = np.trace(emp_cov) / n_features
      shrunk_cov = (1.0 - shrinkage) * emp_cov
      shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
      return shrunk_cov


    def softmax(slf,X, copy=True):
      if copy:
          X = np.copy(X)
      max_prob = np.max(X, axis=1).reshape((-1, 1))
      X -= max_prob
      np.exp(X, X)
      sum_prob = np.sum(X, axis=1).reshape((-1, 1))
      X /= sum_prob
      return X

    def iterative_A(self,A, max_iterations=3):
      '''
      calculate the largest eigenvalue of A
      '''
      x = A.sum(axis=1)
      #k = 3
      for _ in range(max_iterations):
          temp = np.dot(A, x)
          y = temp / np.linalg.norm(temp, 2)
          temp = np.dot(A, y)
          x = temp / np.linalg.norm(temp, 2)
      return np.dot(np.dot(x.T, A), y)
    
    def _solve_eigen2(self, X, y, shrinkage):

        U,S,Vt = np.linalg.svd(np.float32(X), full_matrices=False)


        # solve Ax = b for the best possible approximate solution in terms of least squares
        self.x_hat2 = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

        y_pred1=X@self.x_hat1
        y_pred2=X@self.x_hat2

        scores_c = -np.mean((y_pred2 - y)**2)
        return scores_c,
    def _solve_eigen(self, X, y, shrinkage):

        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
      
        # X_ = pairwise_kernels(X, metric='linear')
        X_=X
       
        means = np.zeros(shape=(len(classes), X_.shape[1]))
        np.add.at(means, y, X_)
        means /= cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X_.shape[1], X_.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X_[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(self._cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = self.iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        # print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = self._cov(X_, shrinkage=self.shrinkage) 

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter
        # print(shrunk_Sw)
        # evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        # print(np.linalg.inv(shrunk_Sw))
        
        evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw)@Sb)
        
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        self.idx=np.argsort(evals)[0:len(X)//2]
   
 
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )


    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N
        '''
        # X,y,y_reg=self.sample_based_on_classes(X,y,y_reg)
        self.classes_ = np.unique(y)
        #n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage,)

        return self
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new #[:, : self._max_components]
   
        
    def energy_score(self, logits):
        logits = to_torch(logits)
        return torch.logsumexp(logits, dim=-1).numpy()

    def predict_proba(self, X,y):

        logits = np.dot(X, self.coef_.T) + self.intercept_
        scores = self.softmax(logits)
        return scores 

    def sample_based_on_classes(self,X,y,y_reg):
        import random
        X_new=[]
        y_new=[]

        labels=np.unique(y)
        mean_labels=np.zeros(len(labels))
        for label in labels:
          idx=np.where(y==label)
          X_label=X[idx]
          y_label=y[idx]
          y_label_reg=y_reg[idx]
          mean_labels[label]=np.mean(X_label)

        for label in labels:
          idx=np.where(y==label)
          X_label=X[idx]
          y_label=y[idx]
          y_label_reg=y_reg[idx]
          mean_label=np.mean(X_label)
          dist=0
          for label_ in labels:
            if label==label_:
              continue
            dist+=np.linalg.norm(X_label-mean_labels[label_],axis=-1)
          idx=np.argsort(dist)[len(X_label)//3:2*len(X_label)//3]
          if label==0:
            X_new=X_label[idx]
            y_new=y_label[idx]
            y_new_reg=y_label_reg[idx]
          else:
            X_new=np.append(X_new,X_label[idx],axis=0)
            y_new=np.append(y_new,y_label[idx],axis=0)
            y_new_reg=np.append(y_new_reg,y_label_reg[idx],axis=0)
        idx=np.arange(len(X_new))
        random.shuffle(idx)
        return X_new[idx],y_new[idx],y_new_reg[idx]



def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        # k = min(N, D)
        N, D = f.shape  

        # direct SVD may be expensive
        if N > D: 
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k, s.shape = k, vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            
            # x has shape [k, 1], but actually x should have shape [N, 1]
            x = u.T @ y_  
            x2 = x ** 2
            # if k < N, we compute sum of xi for 0 singular values directly
            res_x2 = (y_ ** 2).sum() - x2.sum()  

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point
    # _fit = _fit_icml

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels

        :return: LogME score (how well f can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  
        # return np.argmax(logits, axis=-1)
        return prob


def LEEP(X, y, model_name='resnet50'):

    n = len(y)
    num_classes = len(np.unique(y))

    # read classifier
    # Group1: model_name, fc_name, model_ckpt
    ckpt_models = {
        'densenet121': ['classifier.weight', './models/group1/checkpoints/densenet121-a639ec97.pth'],
        'densenet169': ['classifier.weight', './models/group1/checkpoints/densenet169-b2777c0a.pth'],
        'densenet201': ['classifier.weight', './models/group1/checkpoints/densenet201-c1103571.pth'],
        'resnet34': ['fc.weight', './models/group1/checkpoints/resnet34-333f7ec4.pth'],
        'resnet50': ['fc.weight', './models/group1/checkpoints/resnet50-19c8e357.pth'],
        'resnet101': ['fc.weight', './models/group1/checkpoints/resnet101-5d3b4d8f.pth'],
        'resnet152': ['fc.weight', './models/group1/checkpoints/resnet152-b121ed2d.pth'],
        'mnasnet1_0': ['classifier.1.weight', './models/group1/checkpoints/mnasnet1.0_top1_73.512-f206786ef8.pth'],
        'mobilenet_v2': ['classifier.1.weight', './models/group1/checkpoints/mobilenet_v2-b0353104.pth'],
        'googlenet': ['fc.weight', './models/group1/checkpoints/googlenet-1378be20.pth'],
        'inception_v3': ['fc.weight', './models/group1/checkpoints/inception_v3_google-1a9a5a14.pth'],
    }
    ckpt_loc = ckpt_models[model_name][1]
    fc_weight = ckpt_models[model_name][0]
    fc_bias = fc_weight.replace('weight', 'bias')
    ckpt = torch.load(ckpt_loc, map_location='cpu')
    fc_weight = ckpt[fc_weight].detach().numpy()
    fc_bias = ckpt[fc_bias].detach().numpy()

    # p(z|x), z is source label
    prob = np.dot(X, fc_weight.T) + fc_bias
    prob = softmax(prob)   # p(z|x), N x C(source)

    pyz = np.zeros((num_classes, 1000))  # C(source) = 1000
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n
    
    pz = np.sum(pyz, axis=0)     # marginal probability
    py_z = pyz / pz              # conditional probability, C x C(source)
    py_x = np.dot(prob, py_z.T)  # N x C

    # leep = E[p(y|x)]
    leep_score = np.sum(py_x[np.arange(n), y]) / n
    return leep_score


def NLEEP(X, y, component_ratio=5):
    print(1)
    n = len(y)
    num_classes = len(np.unique(y))
    # PCA: keep 80% energy
    pca_80 = PCA(n_components=0.8)
    pca_80.fit(X)
    X_pca_80 = pca_80.transform(X)
    print(2)
    # GMM: n_components = component_ratio * class number
    n_components_num = component_ratio * num_classes
    gmm = GaussianMixture(n_components= n_components_num).fit(X_pca_80)
    prob = gmm.predict_proba(X_pca_80)  # p(z|x)
    
    # NLEEP
    pyz = np.zeros((num_classes, n_components_num))
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n   
    pz = np.sum(pyz, axis=0)    
    py_z = pyz / pz             
    py_x = np.dot(prob, py_z.T) 
    print(3)
    # nleep_score
    nleep_score = np.sum(py_x[np.arange(n), y]) / n
    return nleep_score


def LogME_Score(X, y):

    logme = LogME(regression=False)
    score = logme.fit(X, y)
    return score


def SFDA_Score(X, y):
    starttime_opt = time.time()
    n = len(y)
    num_classes = len(np.unique(y))
    
    SFDA_first = SFDA()
    prob = SFDA_first.fit(X, y).predict_proba(X)  # p(y|x)
    
    # soften the probability using softmax for meaningful confidential mixture
    prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True) 
    means, means_ = _class_means(X, y)  # class means, outer classes means
    
    # ConfMix
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        y_prob = np.take(prob, indices, axis=0)
        y_prob = y_prob[:, y_]  # probability of correctly classifying x with label y        
        X[indices] = y_prob.reshape(len(y_prob), 1) * X[indices] + \
                            (1 - y_prob.reshape(len(y_prob), 1)) * means_[y_]
    
    SFDA_second = SFDA(shrinkage=SFDA_first.shrinkage)
    prob = SFDA_second.fit(X, y).predict_proba(X)   # n * num_cls

    # leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
    sfda_score = np.sum(prob[np.arange(n), y]) / n
    return sfda_score
    # return prob


def LDA_Score(X,y):

  n = len(y)
  num_classes = len(np.unique(y))


  prob = LDA().fit(X, y).predict_proba(X,y)  # p(y|x)
  n = len(y)
  # # # ## leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
  lda_score = np.sum(prob[np.arange(n), y]) / n
 
  return lda_score
  
  
  
def KFDA_Score(X, y):

    n = len(y)
    num_classes = len(np.unique(y))
    
    SFDA_first = kernel_FDA()
    sfda_score = SFDA_first.fit(X, y)# p(y|x)
    
    return sfda_score

def MyFDA_Score(X, y):

    n = len(y)
    num_classes = len(np.unique(y))
    
    SFDA_first = My_FDA()
    sfda_score = SFDA_first.fit(X, y)# p(y|x)
    
    return sfda_score

def Energy_Score(logits,percent,tail):
    logits = to_torch(logits)
    energy_score=torch.logsumexp(logits, dim=-1).numpy()
    if tail=='bot':
        chs = list(np.argsort(energy_score)[0:int(percent*10)*len(energy_score)//1000]) # #
    else:
        chs = list(np.argsort(energy_score)[-int(percent*10)*len(energy_score)//1000:])
    energy_score = energy_score[chs].mean() 
    return energy_score

def PARC_Score(X, y, ratio=2):
    
    num_sample, feature_dim = X.shape
    ndims = 32 if ratio > 1 else int(feature_dim * ratio)  # feature reduction dimension

    if num_sample > 15000:
        from utils_cr import initLabeled
        p = 15000.0 / num_sample
        labeled_index = initLabeled(y, p=p)
        features = X[labeled_index]
        targets = X[labeled_index]
        print("data are sampled to {}".format(features.shape))

    method = PARC(n_dims = ndims)
    parc_score = method(features=X, y=y)

    return parc_score


def one_hot(a, nclass=None): 
    if nclass is None: 
        nclass = a.max()+1 
    b = np.zeros((a.size, nclass))
    b[np.arange(a.size), a] = 1.
    return b

# def PAC_Score(features_np_all, label_np_all,
#                         lda_factor, label_mode='categorical',reg_label=None,reg_cl=False,max_iter=1):
#   """Compute the PAC_Gauss score with diagonal variance."""
#   starttime = time.time()
#   if label_mode == 'categorical': 
#     label_np_all = one_hot(label_np_all)  # [n, v]
#   nclasses = label_np_all.shape[-1]  
#   np.random.seed(2)

#   mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
#   features_np_all -= mean_feature  # [n,k]

#   bs = features_np_all.shape[0]
#   kd = features_np_all.shape[-1] * nclasses
#   ldas2 = lda_factor * bs  # * features_np_all.shape[-1]
#   dinv = 1. / float(features_np_all.shape[-1])


#   # optimizing log lik + log prior
#   def pac_loss_fn(theta):
    
#     nclasses_=nclasses
    
#     theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses_])
    
      
    
#     w = theta[:features_np_all.shape[-1], ]
#     b = theta[features_np_all.shape[-1]:, ]
#     logits = np.matmul(features_np_all, w) + b

    
#     log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)

#     logits_c=log_qz[:,:nclasses]
    

  
#     xent = np.sum(np.sum(
#         label_np_all * (np.log(label_np_all + 1e-10) - logits_c), axis=-1)) / bs 
    
#     loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
   
#     return loss

#   def clas_loss_fn(theta):
    
#     nclasses_=nclasses
    
#     theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses_])
    
      
    
#     w = theta[:features_np_all.shape[-1], ]
#     b = theta[features_np_all.shape[-1]:, ]
#     logits = np.matmul(features_np_all, w) + b

    
#     log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)

#     logits_c=log_qz
    

#     xent = np.sum(np.sum(
#         label_np_all * (np.log(label_np_all + 1e-10) - logits_c), axis=-1)) / bs 
    
#     loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
    
#     return loss

#   # gradient of xent + l2
#   def pac_grad_fn(theta):
#     nclasses_=nclasses
#     if reg_cl:
#       nclasses_=nclasses+2
#     theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses_])
    
#     w = theta[:features_np_all.shape[-1], :]
#     b = theta[features_np_all.shape[-1]:, :]
#     logits = np.matmul(features_np_all, w) + b
    
#     grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
    
#     grad_f[:,:2] -= label_np_all
#     if reg_cl:
#       grad_f[:,2:] = 2*(logits[:,2:]-reg_label)*0.01

    
#     grad_f /= bs
    
#     grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
    
#     grad_w += w / ldas2
    
#     grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
    
#     grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
    
#     return grad

#   # 2nd gradient of theta (elementwise)
#   def pac_grad2(theta):
#     nclasses_=nclasses
#     if reg_cl:
#       nclasses_=nclasses+2
#     theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses_])
    
#     w = theta[:features_np_all.shape[-1], :]
#     b = theta[features_np_all.shape[-1]:, :]
#     logits = np.matmul(features_np_all, w) + b

#     prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
#     grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
#     xx = np.square(features_np_all)  # [n, d]
    
#     grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
#     grad2_w += 1. / ldas2
#     grad2_w[:,2:]=0
#     grad2_f[:,2:]=0
#     grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
#     grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
    
#     return grad2

#   # gradient of xent + l2
#   def clas_grad_fn(theta):
#     nclasses_=nclasses
    
#     theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses_])
    
#     w = theta[:features_np_all.shape[-1], :]
#     b = theta[features_np_all.shape[-1]:, :]
#     logits = np.matmul(features_np_all, w) + b
    
#     grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
    
#     grad_f -= label_np_all

#     grad_f /= bs
    
#     grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
    

#     grad_w += w / ldas2
    
#     grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
    
#     grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
    
#     return grad

#   # 2nd gradient of theta (elementwise)
#   def clas_grad2(theta):
    
#     nclasses_=nclasses
#     theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses_])
    
#     w = theta[:features_np_all.shape[-1], :]
#     b = theta[features_np_all.shape[-1]:, :]
    
#     logits = np.matmul(features_np_all, w) + b
    
#     prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
#     grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
#     xx = np.square(features_np_all)  # [n, d]
    
#     grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
#     grad2_w += 1. / ldas2
#     grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
#     grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
    
#     return grad2


#   nclasses_=nclasses

#   kernel_shape = [features_np_all.shape[-1], nclasses_]
#   theta = np.random.normal(size=kernel_shape) * 0.03
#   theta_1d1 = np.ravel(np.concatenate(
#       [theta, np.zeros([1, nclasses_])], axis=0))

  
#   theta_1d1 = scipy.optimize.minimize(
#       clas_loss_fn, theta_1d1, method="L-BFGS-B",
#       jac=clas_grad_fn,
#       options=dict(maxiter=max_iter), tol=1e-6).x
  
#   pac_opt = clas_loss_fn(theta_1d1)

#   endtime_opt = time.time()
 
#   h = clas_grad2(theta_1d1)
 
#   sigma2_inv = np.sum(h) * ldas2  / kd + 1e-10
#   endtime = time.time()

#   if lda_factor == 10.:
#     s2s = [1000., 100.]
#   elif lda_factor == 1.:
#     s2s = [100., 10.]
#   elif lda_factor == 0.1:
#     s2s = [10., 1.]
    
#   returnv = []

#   for s2_factor in s2s:
#     s2 = s2_factor * dinv
#     pac_gauss = pac_opt + 0.5 * kd / ldas2 * s2 * np.log(
#         sigma2_inv)


#   # the first item is the pac_gauss metric
#   # the second item is the linear metric (without trH)
#   returnv += [("pac_gauss_%.1f" % lda_factor, pac_gauss),
#               ("time", endtime - starttime),
#               ("pac_opt_%.1f" % lda_factor,pac_opt),
#               ("time", endtime_opt - starttime)]

#   return returnv, theta_1d1


def PAC_Score(features_np_all, label_np_all,
                        lda_factor):
  """Compute the PAC_Gauss score with diagonal variance."""
  starttime = time.time()
  nclasses = label_np_all.max()+1
  label_np_all = one_hot(label_np_all)  # [n, v]
  
  mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
  features_np_all -= mean_feature  # [n,k]

  bs = features_np_all.shape[0]
  kd = features_np_all.shape[-1] * nclasses
  ldas2 = lda_factor * bs  # * features_np_all.shape[-1]
  dinv = 1. / float(features_np_all.shape[-1])

  # optimizing log lik + log prior
  def pac_loss_fn(theta):
    theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

    w = theta[:features_np_all.shape[-1], :]
    b = theta[features_np_all.shape[-1]:, :]
    logits = np.matmul(features_np_all, w) + b

    log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    xent = np.sum(np.sum(
        label_np_all * (np.log(label_np_all + 1e-10) - log_qz), axis=-1)) / bs
    loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
    return loss

  # gradient of xent + l2
  def pac_grad_fn(theta):
    theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

    w = theta[:features_np_all.shape[-1], :]
    b = theta[features_np_all.shape[-1]:, :]
    logits = np.matmul(features_np_all, w) + b

    grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
    grad_f -= label_np_all
    grad_f /= bs
    grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
    grad_w += w / ldas2

    grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
    grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
    return grad

  # 2nd gradient of theta (elementwise)
  def pac_grad2(theta):
    theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

    w = theta[:features_np_all.shape[-1], :]
    b = theta[features_np_all.shape[-1]:, :]
    logits = np.matmul(features_np_all, w) + b

    prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
    grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
    xx = np.square(features_np_all)  # [n, d]

    grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
    grad2_w += 1. / ldas2
    grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
    grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
    return grad2

  kernel_shape = [features_np_all.shape[-1], nclasses]
  theta = np.random.normal(size=kernel_shape) * 0.03
  theta_1d = np.ravel(np.concatenate(
      [theta, np.zeros([1, nclasses])], axis=0))

  theta_1d = scipy.optimize.minimize(
      pac_loss_fn, theta_1d, method="L-BFGS-B",
      jac=pac_grad_fn,
      options=dict(maxiter=10), tol=1e-6).x

  pac_opt = pac_loss_fn(theta_1d)
  endtime_opt = time.time()

  h = pac_grad2(theta_1d)
  sigma2_inv = np.sum(h) * ldas2  / kd + 1e-10
  endtime = time.time()

  if lda_factor == 10.:
    s2s = [1000., 100.]
  elif lda_factor == 1.:
    s2s = [100., 10.]
  elif lda_factor == 0.1:
    s2s = [10., 1.]
    
  returnv = []
  for s2_factor in s2s:
    s2 = s2_factor * dinv
    pac_gauss = pac_opt + 0.5 * kd / ldas2 * s2 * np.log(
        sigma2_inv)
    
    # the first item is the pac_gauss metric
    # the second item is the linear metric (without trH)
    returnv += [("pac_gauss_%.1f" % lda_factor, pac_gauss),
                ("time", endtime - starttime),
                ("pac_opt_%.1f" % lda_factor, pac_opt),
                ("time", endtime_opt - starttime)]
  return returnv, theta_1d