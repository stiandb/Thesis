from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from copy import deepcopy

class AutoEncoder(Utils):
	def __init__(self,U_1=None,U_2=None,n_qubits=None,n_weights=None,shots=1000,initial_state= identity_circuit,seed_simulator=None):
		"""
		Inputs:
			
		"""
		self.w_opt = None
		self.first_run = True
		self.n_qubits=n_qubits
		self.loss_fn = UnitaryComparison(U_1,U_2,n_qubits,shots=shots,seed_simulator=seed_simulator)
		self.loss_train = []
		self.n_weights = n_weights
		self.initial_state = initial_state
		

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
		cost_train = self.loss_fn(w,X,self.initial_state)
		if not self.first_run and (cost_train < np.min(np.array(self.loss_train))):
			self.w_opt = w.copy()
		if print_loss:
			print('Training loss: ',cost_train)
		self.loss_train.append(cost_train)
		self.first_run = False
		return(cost_train)
