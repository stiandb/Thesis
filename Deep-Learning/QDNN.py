import numpy as np
import qiskit as qk
import sys
sys.path.append('../')
from dl_utils import *
from scipy.optimize import minimize
from utils import *


class QDNN(Utils):
	def __init__(self,layers=None,loss_fn = None):
		self.layers = layers
		self.outputs = None
		self.loss_fn = loss_fn

	def forward(self,x):
		out = [x]
		for layer in self.layers:
			x = layer(x)
			out.append(x)
		self.outputs = out
		return(x)

	def fit(self,X=None,y=None,method='Cobyla',max_iters = 10000):
		n_inputs = 0
		for layer in self.layers:
			n_inputs += layer.w_size
		w = np.random.randn(n_inputs)
		w = minimize(self.calculate_loss,w,args=(X,y),method=method,options={'disp':True,'maxiter':max_iters}).x
		self.set_weights(w)
		self.loss = np.array(self.loss)
		return(w)

	def calculate_loss(self,w,X,y):
		self.set_weights(w)
		self.loss = []
		y_pred = self.predict(X)
		cost = self.loss_fn(y_pred,y)
		self.loss.append(cost)
		print('Loss: ',cost)
		return(cost)

	def predict(self,X):
		y_pred = []
		for i in range(X.shape[0]):
			x = X[i,:]
			x = x/np.sqrt(np.sum(x**2))
			forward = self.forward(x)
			if np.array_equal(forward,np.array([0 for i in range(self.layers[-1].n_outputs)])) or \
			 np.array_equal(forward,np.array([1 for i in range(self.layers[-1].n_outputs)])):
				forward = np.array([1/3 for i in range(self.layers[-1].n_outputs)])
			else:
				forward /= np.sum(forward)
			y_pred.append(forward)
		y_pred = np.array(y_pred)
		return(y_pred)















