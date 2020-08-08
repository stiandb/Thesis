import numpy as np 
import sys
sys.path.append('../')
from utils import *
from sklearn.metrics import log_loss
from copy import deepcopy

class MSE:
	"""
	Calculated Mean Squared Error loss
	"""
	def __call__(self,y_pred,y):
		"""
		Input:
			y_pred (numpy array) - the predicted values
			y (numpy array) - The actual values
		Output:
			MSE (float)
		"""
		return(np.mean((y_pred.flatten() - y.flatten())**2))

class binary_cross_entropy:
	"""
	Calculates the cross entropy loss for binary classification
	"""
	def __call__(self,y_pred,y):
		"""
		Input:
			y_pred (numpy array) - the predicted values
			y (numpy array) - The actual values
		Output:
			binary cross entropy (float)
		"""
		y_p = y_pred.copy()
		y_p[y_p == 1] = 1 - 1e-14
		y_p[y_p == 0] = 1e-14
		return( np.mean(-np.log(y_p)*y - np.log(1 - y_p)*(1 - y) ))

class cross_entropy:
	"""
	calculates the cross entropy loss for multiple classification models
	"""
	def __call__(self,y_pred,y):
		"""
		Input:
			y_pred (numpy array) - the predicted values
			y (numpy array) - The actual values
		Output:
			cross entropy (float)
		"""
		return(log_loss(y,y_pred))

class rayleigh_quotient:
	"""
	Calculates the Rayleigh quotient for matrix H
	"""
	def __init__(self,H):
		"""
		Input:
			H (numpy 2d array) - The matrix to calculate rayleigh quotient for
		"""
		self.H = H

	def __call__(self,x,*args):
		"""
		Input:
			x (numpy array) - the vector to calculate the Rayleigh quotient for
		Output:
			Rayleigh quotient (float)
		"""
		H = self.H
		x = x.T
		x -= 0.5
		return((x.T@H@x/(x.T@x)).flatten()[0])






class UnitaryComparison:
	"""
	Calculates the squared inner product between two operators, U_1 and U_2
	"""
	def __init__(self,U_1,U_2,n_qubits=None,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,seed_transpiler=None,transpile=False,optimization_level=1,error_mitigator=None):
		"""
		Input:
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
		"""
		self.U_1 = U_1
		self.U_2 = U_2
		self.n_qubits=n_qubits
		self.shots=shots
		self.seed_simulator=seed_simulator
		self.backend=backend
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.coupling_map=coupling_map
		self.error_mitigator = error_mitigator
	def __call__(self,theta_1,theta_2,initial_state):
		"""
		Input:
			theta_1 (numpy 1d array) - The array containing parameters for U_1
			theta_2 (numpy 1d array) - The array containing parameters for U_2
			initial_state (functional) - initial_state(circuit,registers) returns
										circuit,registers with applied initial state on
		Output:
			negative of the squared inner product between state produced by U_1 and U_2
		"""
		shots = self.shots
		seed_simulator=self.seed_simulator
		backend=self.backend
		noise_model=self.noise_model
		basis_gates = self.basis_gates
		transpile=self.transpile
		seed_transpiler=self.seed_transpiler
		optimization_level=self.optimization_level
		coupling_map=self.coupling_map
		error_mitigator = self.error_mitigator
		n_qubits = self.n_qubits
		q_reg = qk.QuantumRegister(n_qubits,'q')
		c_reg = qk.ClassicalRegister(1)
		circuit = qk.QuantumCircuit(q_reg,c_reg)
		registers = [q_reg,c_reg]
		circuit,registers = initial_state(circuit,registers,inverse=False)
		circuit,registers = deepcopy(self.U_1)(theta_1,circuit,registers)
		circuit,registers = deepcopy(self.U_2)(theta_2,circuit,registers)
		circuit,registers = initial_state(circuit,registers,inverse=True)
		ancilla_register = qk.QuantumRegister(1)
		circuit.add_register(ancilla_register)
		registers.insert(1,ancilla_register)
		for i in range(len(registers[0])):
			circuit.x(registers[0][i])
		circuit.mcrx(np.pi,[registers[0][i] for i in range(len(registers[0]))],ancilla_register[0])
		circuit.measure(ancilla_register,registers[-1])
		if transpile:
			circuit = qk.compiler.transpile(circuit,backend=backend,backend_properties=backend.properties(),seed_transpiler=seed_transpiler,optimization_level=optimization_level,basis_gates=basis_gates,coupling_map=coupling_map)
			initial_layout = circuit._layout.get_virtual_bits()
		if noise_model is None:
			job = qk.execute(circuit, backend = backend,seed_simulator=seed_simulator,shots=shots,basis_gates=basis_gates,coupling_map=coupling_map)
			try:
				if not job.error_message() is None:
					print(job.error_message())
			except:
				None
			job = job.result()
		else:
			job = qk.execute(circuit, backend = backend,seed_simulator=seed_simulator,shots=shots,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map).result()
		if not error_mitigator is None:
			try:
				n_qubits = circuit.n_qubits
			except:
				n_qubits = circuit.num_qubits
			qubit_list = [-1]
			meas_filter = error_mitigator(n_qubits,qubit_list,backend,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,shots=shots,transpile=transpile,seed_transpiler=seed_transpiler,optimization_level=optimization_level,initial_layout=initial_layout)
			result = meas_filter.apply(job)
			result = result.get_counts(0)
		else:
			result = job.get_counts(circuit)
		inner_product = 0
		for key,value in result.items():
			if key == '1':
				inner_product += value
		inner_product /= self.shots
		del circuit,registers,ancilla_register
		return(-inner_product)








