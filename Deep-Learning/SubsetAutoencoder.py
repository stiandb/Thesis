import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from copy import deepcopy

class SubsetAutoencoder(Utils):
	def __init__(self,k = None,layers=None,U_subenc=None,k_encoders=False):
		"""
		Inputs:
			layers (list) - List containing layers for dense neural network
			loss_fn (callable) - loss_fn(y_pred, y) returns the loss where y_pred is 
									the prediction and y is the actual target variable
		"""
		self.layers = layers if not k_encoders else [deepcopy(layers) for i in range(k)]
		self.k = k
		self.w_opt = None
		self.first_run = True
		self.U_subenc = U_subenc
		self.loss_fn = SubsetAutoEncoderInnerProduct(self.U_subenc)
		self.loss_train = []
		self.k_encoders = k_encoders

	def forward(self,X):
		"""
		Input:
			x (numpy array) - Numpy array of dimension [n,p], where n is the number of samples 
								and p is the number of predictors
		Output:
			x (numpy array) - The output from the neural network.
		"""

		params= []
		X = np.split(X,self.k,axis=1)
		for k,x in enumerate(X):
			if self.k_encoders:
				layer_list = self.layers[k]
			else:
				layer_list = self.layers
			for sub_layer in layer_list:
				x = sub_layer(x)
			params.append(x)
		params = np.hstack(params)
		return(params)

	def fit(self,X,method='Powell',max_iters=1000,print_loss=False):
		"""
		Uses classical optimization to train the neural network.
		Input:
			X (numpy array) - design matrix for the problem
			y (numpy array) - target variable for the problem
			method (str)    - the classical optimization method to use
			max_inters (int)- The maximum number of iterations for the classical
								optimization.
		"""
		options = {'disp':True,'maxiter':max_iters}
		self.n_weights = 0
		for layer in self.layers:
			if type(layer) is list:
				for sub_layer in layer:
					self.n_weights += sub_layer.w_size
			else:
				self.n_weights += layer.w_size
		w = 1+0.1*np.random.randn(self.n_weights)
		w = minimize(self.calculate_loss,w,args=(X,print_loss),method=method,options=options).x
		self.loss_train = np.array(self.loss_train)
		return(w)
		

	def calculate_loss(self,w,X,print_loss=False):
		"""
		Input:
			w (numpy array) - One dimensional array containing 
								all network weights
			X (numpy array) - design matrix for the problem
			y (numpy array) - target variable for the problem
		Output:
			cost (float) 	- The loss for the data.
		"""
		n_inputs = X.shape[1]
		self.set_weights(w)
		theta = self.forward(X)
		n_qubits = int(np.ceil(np.log2(n_inputs)))
		amp_reg = qk.QuantumRegister(n_qubits)
		cl_reg = qk.ClassicalRegister(1)
		circuit = qk.QuantumCircuit(amp_reg,cl_reg)
		registers = [amp_reg,cl_reg]
		cost_train = self.loss_fn(2*np.pi*theta,X,circuit,registers)
		if not self.first_run and (cost_train < np.min(np.array(self.loss_train))):
			self.w_opt = w.copy()
		if print_loss:
			print('Training loss: ',cost_train)
		self.loss_train.append(cost_train)
		self.first_run = False
		return(cost_train)





