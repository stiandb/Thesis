import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from hamiltonian import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
import matplotlib.pylab as plt

class SubsetAutoencoder:
	def __init__(self,k = None,layers=None,U_subenc=None):
		"""
		Inputs:
			layers (list) - List containing layers for dense neural network
			loss_fn (callable) - loss_fn(y_pred, y) returns the loss where y_pred is 
									the prediction and y is the actual target variable
		"""
		self.layers = layers
		self.k = k
		self.w_opt = None
		self.first_run = True
		self.U_subenc = U_subenc

	def forward(self,X):
		"""
		Input:
			x (numpy array) - Numpy array of dimension [n,p], where n is the number of samples 
								and p is the number of predictors
		Output:
			x (numpy array) - The output from the neural network.
		"""

		params= []
		X = np.split(X,self.k)
		for x in X:
			x = x.reshape(1,x.shape[0])
			for layer in self.layers:
				if type(layer) is list:
					x_ = []
					idx=0
					for sub_layer in layer:
						x_.append(sub_layer(x[:,idx:idx+sub_layer.n_qubits]))
						idx += sub_layer.n_qubits
					x = np.hstack(x_)
				else:
					x = layer(x)
			params.append(x)

		return(np.hstack(params))

	def fit(self,X,y,method='Powell',print_loss=False):
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
		w = minimize(self.calculate_loss,w,args=(X,y,X_val,y_val,print_loss),method=method,options=options,tol=tol).x
		self.set_weights(self.w_opt)
		self.loss_train = np.array(self.loss_train)
		

	def calculate_loss(self,w,X,y,print_loss=False):
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
		if not self.first_run and (cost_train < np.min(np.array(self.loss_train))):
			self.w_opt = w.copy()
		if print_loss:
			print('Training loss: ',cost_train)
		self.loss_train.append(cost_train)
		self.first_run = False
		return(cost_train)
np.random.seed(7)

X = np.random.randn(16)
layer = GeneralLinear(n_qubits=2,n_outputs=3*2,n_weights_a=3*2*2,n_weights_ent=1,U_enc=AmplitudeEncoder(),U_a=y_rotation_ansatz,U_ent=EntanglementRotation(bias=True),shots=100,seed_simulator=42)





