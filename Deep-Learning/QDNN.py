import sys
sys.path.append('../')
from dl_utils import *
from scipy.optimize import minimize
from utils import *
import numpy as np
import qiskit as qk


class QDNN(Utils):
	def __init__(self,layers=None,loss_fn = None,classification=False):
		"""
		Inputs:
			layers (list) - List containing layers for dense neural network
			loss_fn (callable) - loss_fn(y_pred, y) returns the loss where y_pred is 
									the prediction and y is the actual target variable
		"""
		self.classification=classification
		self.layers = layers
		self.loss_fn = loss_fn
		self.loss_train = []
		self.loss_val = []
		self.w_opt = None
		self.first_run = True

	def forward(self,X):
		"""
		Input:
			x (numpy array) - Numpy array of dimension [n,p], where n is the number of samples 
								and p is the number of predictors
		Output:
			x (numpy array) - The output from the neural network.
		"""
		for layer in self.layers:
			if type(layer) is list:
				X_ = []
				idx=0
				for sub_layer in layer:
					X_.append(sub_layer(X[:,idx:idx+sub_layer.n_qubits]))
					idx += sub_layer.n_qubits
				X = np.hstack(X_)
			else:
				X = layer(X)
		if self.classification:
			X = X/(np.sum(X,axis=1).reshape(X.shape[0],1) + 1e-14)
		return(X)

	def fit(self,X,y,X_val=None,y_val=None,method='Powell',max_iters = 1000,max_fev=None,tol=1e-14,seed=None):
		"""
		Uses classical optimization to train the neural network.
		Input:
			X (numpy array) - design matrix for the problem
			y (numpy array) - target variable for the problem
			method (str)    - the classical optimization method to use
			max_inters (int)- The maximum number of iterations for the classical
								optimization.
		"""
		if not seed is None:
			np.random.seed(seed)
		if not max_fev is None:
			options = {'disp':True,'maxiter':max_iters,'maxfev':max_fev}
		else:
			options = {'disp':True,'maxiter':max_iters}
		self.n_weights = 0
		for layer in self.layers:
			if type(layer) is list:
				for sub_layer in layer:
					self.n_weights += sub_layer.w_size
			else:
				self.n_weights += layer.w_size
		w = np.random.randn(self.n_weights)
		w = minimize(self.calculate_loss,w,args=(X,y,X_val,y_val),method=method,options=options,tol=tol).x
		self.set_weights(self.w_opt)
		self.loss_train = np.array(self.loss_train)
		self.loss_val = np.array(self.loss_val)

	def calculate_loss(self,w,X,y,X_val=None,y_val=None):
		"""
		Input:
			w (numpy array) - One dimensional array containing 
								all network weights
			X (numpy array) - design matrix for the problem
			y (numpy array) - target variable for the problem
		Output:
			cost (float) 	- The loss for the data.
		"""
		self.set_weights(w)
		y_pred = self.forward(X)
		cost_train = self.loss_fn(y_pred,y)
		if X_val is None:
			if not self.first_run and (cost_train < np.min(np.array(self.loss_train))):
				self.w_opt = w.copy()
			print('Training loss: ',cost_train)
		else:
			y_val_pred = self.forward(X_val)
			cost_val = self.loss_fn(y_val_pred,y_val)
			if not self.first_run and (cost_val < np.min(np.array(self.loss_val))):
				self.w_opt = w.copy()
			print('Training loss: ',cost_train, ' Validation loss: ',cost_val)
			self.loss_val.append(cost_val)
		self.loss_train.append(cost_train)
		self.first_run = False
		return(cost_train)















