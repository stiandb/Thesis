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

	def forward(self,X):
		"""
		Input:
			x (numpy array) - Numpy array of dimension [n,p], where n is the number of samples 
								and p is the number of predictors
		Output:
			x (numpy array) - The output from the neural network.
		"""
		for layer in self.layers:
			X = layer(X)
		if self.classification:
			X = X/(np.sum(X,axis=1).reshape(X.shape[0],1) + 1e-14)
		return(X)

	def fit(self,X,y,method='Powell',max_iters = 10000):
		"""
		Uses classical optimization to train the neural network.
		Input:
			X (numpy array) - design matrix for the problem
			y (numpy array) - target variable for the problem
			method (str)    - the classical optimization method to use
			max_inters (int)- The maximum number of iterations for the classical
								optimization.
		"""
		self.n_weights = 0
		for layer in self.layers:
			self.n_weights += layer.w_size
		w = np.random.randn(self.n_weights)
		w = minimize(self.calculate_loss,w,args=(X,y),method=method,options={'disp':True,'maxiter':max_iters}).x
		self.set_weights(w)

	def calculate_loss(self,w,X,y):
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
		cost = self.loss_fn(y_pred,y)
		print('Loss: ',cost)
		return(cost)















