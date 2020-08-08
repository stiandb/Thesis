from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from copy import deepcopy



class AutoEncoder(Utils):
	"""
	Class that can be utilized to maximize or minimize the squared inner product between a state produced by operator U_1
	and operator U_2
	"""
	def __init__(self,U_1=None,U_2=None,n_qubits=None,n_weights=None,shots=1000,initial_state= identity_circuit,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,seed_transpiler=None,transpile=False,optimization_level=1,error_mitigator=None,minimize_inner_product=False):
		"""
		Inputs:
			U_1 (functional) - Accepts theta,circuit,registers, where theta is a 1d numpy array containing parameters for operator.
								ciruit is the qiskit QuantumCircuit to apply the operation on. registers is a list containing the 
								register to apply operation on as first element.
								U_1 is one of the operators to compare with squared inner product
			U_2 (functional) - Same as U_1, but the operator to compare with U_1
			n_weights (int) - The number of weights U_1 are dependent on.
			initial_state (functional) - Functional that puts qubits into initial state for AutoEncoder
			backend - The qiskit backend.
			seed_simulator (int or None) - The seed to be utilized when simulating quantum computer
			noise_model - The qiskit noise model to utilize when simulating noise.
			basis_gates - The qiskit basis gates allowed to utilize
			coupling_map - The coupling map which explains the connection between each qubit
			shots (int) - How many times to measure circuit
			transpile (boolean) - If True, transpiler is used
			seed_transpiler (int) - The seed to use for the transoiler
			optimization_level (int) - The optimization level for the transpiler. 0 is no optimization,
										3 is the heaviest optimization
			error_mitigator (functional) - returns the filter to apply for error reduction
			minimize_inner_product (boolean) - If True, The inner product is minimized rather than maximized between the two operator states.
		"""
		self.w_opt = None
		self.first_run = True
		self.n_qubits=n_qubits
		self.loss_fn = UnitaryComparison(U_1,U_2,n_qubits,shots=shots,seed_simulator=seed_simulator,backend=backend,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,transpile=transpile,seed_transpiler=seed_transpiler,optimization_level=optimization_level,error_mitigator=error_mitigator)
		self.loss_train = []
		self.n_weights = n_weights
		self.initial_state = initial_state
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.coupling_map=coupling_map
		self.error_mitigator = error_mitigator
		self.minimize_inner_product=minimize_inner_product
		

	def fit(self,X=0,method='Powell',max_iters=1000,print_loss=False):
		"""
		Uses classical optimization to train the neural network.
		Input:
			X (numpy array) - design matrix for the problem
			y (numpy array) - target variable for the problem
			method (str)    - the classical optimization method to use
			max_iters (int)- The maximum number of iterations for the classical
								optimization.
		Output:
			w (numpy 1d array) - The parameters best maximizes the inner product
								between the state produced by U_1 and U_2
		"""
		options = {'disp':True,'maxiter':max_iters}
		w = 1+0.1*np.random.randn(self.n_weights)
		w = minimize(self.calculate_loss,w,args=(X,print_loss),method=method,options=options).x
		self.loss_train = np.array(self.loss_train)
		return(self.w_opt)
		

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
		if self.minimize_inner_product:
			cost_train = - cost_train
		return(cost_train)
