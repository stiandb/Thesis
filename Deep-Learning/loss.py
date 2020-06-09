import numpy as np 
from sklearn.metrics import log_loss

class MSE:
	def __call__(self,y_pred,y):
		return(np.mean((y_pred.flatten() - y.flatten())**2))

class binary_cross_entropy:
	def __call__(self,y_pred,y):
		y_p = y_pred.copy()
		y_p[y_p == 1] = 1 - 1e-14
		y_p[y_p == 0] = 1e-14
		return( np.mean(-np.log(y_p)*y - np.log(1 - y_p)*(1 - y) ))

class cross_entropy:
	def __call__(self,y_pred,y):
		return(log_loss(y,y_pred))

class rayleigh_quotient:
	def __init__(self,H):
		self.H = H

	def __call__(self,x,*args):
		H = self.H
		x -= 1
		x = x.reshape(x.flatten().shape[0],1)
		return((x.T@H@x/(x.T@x)).flatten()[0])