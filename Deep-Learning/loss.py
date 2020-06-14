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
		x = x.T
		x -= 0.5
		return((x.T@H@x/(x.T@x)).flatten()[0])

class eigenvector_ode:
	def __init__(self,A,dt):
		self.dt = dt
		self.A = A

	def __call__(self,x,*args):
		loss = 0
		x = x[0,:,:]
		for t in range(x.shape[0]-1):
			loss += np.mean(( (x[t+1,:] - x[t,:])/self.dt + self.A@x[t+1,:] - (x[t+1,:].T@self.A@x[t+1,:])*x[t+1,:] )**2)
		quotient = rayleigh_quotient(self.A)
		print('Quotient: ',quotient(x[-1,:]))
		return(loss)

