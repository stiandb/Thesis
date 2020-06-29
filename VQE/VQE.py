import qiskit as qk 
import numpy as np
from scipy.optimize import minimize
from utils import *
from inspect import getfullargspec


class VQE:
	def __init__(self,hamiltonian_list,ansatz,n_qubits,ancilla=0,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,max_energy=False,transpile=False,seed_transpiler=None,optimization_level=1,coupling_map=None,error_mitigator=None):
		"""
		Inputs:
			hamiltonian_list (list) - List containing each term of the hamiltionian
			ansatz (function) - A function which accepts theta,circuit and registers as its arguments.
								theta should be the parameters to solve for, circuit is the quantum circuit
								registers is a list of qiskit registers where the ansatz register is the first entry
								and the classical register is the last.
			n_qubits (int)    - The number of qubits in the ansatz register
			ancilla (int)     - Some ansatzes requires ancilla register. This int specifies the number of ancilla qubits
			shots (int) 	  - Specifies the number of times a circuit is run to evaluate expectation value
			seed  (NoneType or int)       - Seed for circuit measurement.
			max_energy (boolean) 		  - If False (default), the minimum energy is approximated. The maximum
											energy is approximated if True.

		"""
		self.hamiltonian_list = hamiltonian_list
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.ansatz = ansatz
		self.n_qubits = n_qubits
		self.shots = shots
		self.max_energy = max_energy
		self.ancilla=ancilla
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.coupling_map=coupling_map
		spec = getfullargspec(self.ansatz)
		if 'classical_bits' in spec[0]:
			self.set_classical_bits = True
		else:
			self.set_classical_bits = False
		self.energies = []
		self.error_mitigator = error_mitigator




	def expectation_value(self,theta):
		"""
		Calculates expectation values and adds them together
		"""
		E = 0
		circuit,registers = None, None
		for pauli_string in self.hamiltonian_list:
			factor = pauli_string[0]
			if factor == 0:
				continue
			classical_bits = len(pauli_string[1:])
			if classical_bits == 0:
				E += factor
				continue
			if self.set_classical_bits:
				circuit,registers = self.ansatz(theta,circuit,registers,classical_bits)
			else:
				circuit,registers = initialize_circuit(self.n_qubits,self.ancilla,classical_bits)
				circuit,registers = self.ansatz(theta,circuit,registers)
			qubit_list = []
			for qubit,gate in pauli_string[1:]:
				circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
				qubit_list.append(qubit)
			E += measure_expectation_value(qubit_list,factor,circuit,registers,seed_simulator=self.seed_simulator,backend=self.backend,shots=self.shots,noise_model=self.noise_model,basis_gates=self.basis_gates,transpile=self.transpile,optimization_level=self.optimization_level,seed_transpiler=self.seed_transpiler,coupling_map=self.coupling_map,error_mitigator=self.error_mitigator)
			if not self.seed_simulator is None:
				self.seed_simulator += 1
		print('<E> = ', E)
		if self.max_energy:
			E = -E
		self.energies.append(E)
		return(E)

	def classical_optimization(self,theta,method='L-BFGS-B',max_iters = 1000):
		"""
		Performs a classical optimization method to find the optimal parameters.
		This function is used directly after initialization of class.
		Input:
			theta (numpy array) - 1D array with the parameters to optimize
			method (str)		- String specifying the optimization method to use.
								  See scipy.optimize minimize documentation for several options
			max_iters (int) 	- Maximum number of iterations for optimization method
		Output:
			theta (numpy array) - The optimal parameters  
		"""
		if method == 'L-BFGS-B':
			bounds = [(0,2*np.pi) for i in theta]
		else:
			bounds = None
		result = minimize(self.expectation_value,theta,bounds = bounds,method=method,options={'disp':True,'maxiter':max_iters})
		theta = result.x
		return(theta)