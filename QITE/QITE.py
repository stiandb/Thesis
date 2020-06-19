import sys
sys.path.append('../')
from utils import *
from copy import deepcopy
from scipy.optimize import fmin_cg
from sympy import Matrix
import scipy as scipy

class QITE:
	def __init__(self,n_qubits,hamiltonian_list,a_list,dt,initial_state,backend=qk.Aer.get_backend('qasm_simulator'),seed_simulator=None,noise_model=None,shots=1000,lamb=0.1):
		self.hamiltonian_list = hamiltonian_list
		self.n_terms = len(hamiltonian_list)
		self.a_list = a_list
		for i in range(len(self.a_list)):
			for j in range(len(self.a_list[i])):
				self.a_list[i][j][0] = 1
		self.dt = dt
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.shots=shots
		self.n_qubits=n_qubits
		self.initial_state = initial_state
		self.lamb=lamb
		self.time_evolution_list = []
		self.noise_model=noise_model

	def S_ij(self,i,j,term):
		sigma_ij = operator_product(deepcopy(self.a_list[term][i]),deepcopy(self.a_list[term][j]))
		factor = sigma_ij[0]
		qubit_and_gate = sigma_ij[1:]
		classical_bits = len(qubit_and_gate)
		if classical_bits == 0:
			return(factor)
		circuit,registers = initialize_circuit(self.n_qubits,1,classical_bits)
		circuit,registers = self.evolve_state(circuit,registers)
		qubit_list = []
		for qubit,gate in qubit_and_gate:
			qubit_list.append(qubit)
			circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
		expval = measure_expectation_value(qubit_list,factor,circuit,registers,seed_simulator=self.seed_simulator,backend=self.backend,shots=self.shots,noise_model=self.noise_model)
		return(expval)

	def b_i(self,i,term):
		sigma_iH = operator_product(deepcopy(self.a_list[term][i]),deepcopy(self.hamiltonian_list[term]))
		factor = sigma_iH[0]
		classical_bits = len(sigma_iH[1:])
		if classical_bits == 0 or np.imag(factor) == 0:
			return(np.imag(factor))
		circuit, registers = initialize_circuit(self.n_qubits,1,classical_bits)
		circuit, registers = self.evolve_state(circuit,registers)
		qubit_list = []
		for qubit,gate in sigma_iH[1:]:
			circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
			qubit_list.append(qubit)
		expval = measure_expectation_value(qubit_list,factor,circuit,registers,seed_simulator=self.seed_simulator,backend=self.backend,shots=self.shots,noise_model=self.noise_model)
		return(np.imag(expval))

	def b_ii(self,i,term):
		sigma_iH = operator_product(deepcopy(self.a_list[term][i]),deepcopy(self.hamiltonian_list[term]))
		factor = sigma_iH[0]
		circuit, registers = initialize_circuit(self.n_qubits,1,1)
		circuit, registers = self.evolve_state(circuit,registers)
		if np.imag(factor) != 0:
			imag = False
		else:
			imag= True
		res = hadamard_test(circuit,registers,sigma_iH,imag=imag)
		return(res)


	def squared_norm(self,term):
		classical_bits = len(self.hamiltonian_list[term][1:])
		factor = self.hamiltonian_list[term][0]
		if factor == 0:
			return(1)
		if classical_bits == 0:
			return(1 - 2*self.dt*factor)
		circuit,registers = initialize_circuit(self.n_qubits,1,classical_bits)
		circuit,registers = self.evolve_state(circuit,registers)
		qubit_list = []
		for qubit,gate in self.hamiltonian_list[term][1:]:
			circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
			qubit_list.append(qubit)
		expval = measure_expectation_value(qubit_list,factor,circuit,registers,seed_simulator=self.seed_simulator,backend=self.backend,shots=self.shots,noise_model=self.noise_model)
		return(1 - 2*self.dt*expval)

	
	def evolve_state(self,circuit,registers):
		if len(self.time_evolution_list) == 0:
			circuit, registers = self.initial_state(circuit,registers)
			return(circuit,registers)
		circuit,registers = self.initial_state(circuit,registers)
		time_evolution = TimeEvolutionOperator(self.time_evolution_list,self.dt,self.dt)
		circuit,registers = time_evolution(circuit,registers)
		return(circuit,registers)


	def step(self,term):
		S = np.zeros((len(self.a_list[term]),len(self.a_list[term])),dtype=complex)
		b = np.zeros((len(self.a_list[term])),dtype=complex)
		c = self.squared_norm(term)
		for i in range(len(self.a_list[term])):
			b[i] = -(2/np.sqrt(c))*self.b_i(i,term)
			for j in range(i,len(self.a_list[term])):
				S[i,j] = self.S_ij(i,j,term)
		S_temp = S.copy()
		for i in range(S.shape[0]):
			S_temp[i,i] = 0
		S = S + S_temp.conj().T
		dalpha = np.eye(b.shape[0])*self.lamb #regularization
		A = S + S.T + dalpha
		x = scipy.sparse.linalg.cg(A,-b)[0]
		if np.any(np.iscomplex(c)):
			print('Warning: a[m] contains complex number. All solutions should be real')
			print('a[m]: ',x)
		else:
			x = np.real(x)
		for i in range(x.shape[0]):
			self.a_list[term][i][0] = x[i]
		self.time_evolution_list_temp.extend(deepcopy(self.a_list[term]))
		for i in range(x.shape[0]):
			self.a_list[term][i][0] = 1
		

	def solve(self,iterations):
		print('Initial Energy: ', self.measure_energy())
		for iteration in range(iterations):
			self.time_evolution_list_temp = []
			for term in range(len(self.hamiltonian_list)):
				self.step(term)
				print('Hamiltonian term {}/{}.'.format(term+1,len(self.hamiltonian_list)))
			self.time_evolution_list.extend(self.time_evolution_list_temp)
			print('Energy after iteration {}: '.format(iteration+1),self.measure_energy())
		return(self.measure_energy())

	def measure_energy(self):
		expval = 0
		for h_m in self.hamiltonian_list:
			classical_bits = len(h_m[1:])
			factor = h_m[0]
			if factor == 0:
				continue
			if classical_bits == 0:
				expval += factor
				continue
			circuit,registers = initialize_circuit(self.n_qubits,1,classical_bits)
			circuit,registers = self.evolve_state(circuit,registers)
			qubit_list = []
			for qubit,gate in h_m[1:]:
				circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
				qubit_list.append(qubit)
			expval += measure_expectation_value(qubit_list,factor,circuit,registers,seed_simulator=self.seed_simulator,backend=self.backend,shots=self.shots,noise_model=self.noise_model)
		return(expval)


		