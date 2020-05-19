import sys
sys.path.append('../')
from utils import *
from copy import deepcopy

class QATE:
	def __init__(self,n_qubits,H_0,H_1,initial_state,dt,t):
		self.H_0_temp = deepcopy(H_0)
		self.H_1_temp = deepcopy(H_1)
		self.H_0 = H_0
		self.H_1 = H_1
		self.initial_state = initial_state
		self.dt = dt
		self.t = t
		self.iterations = int(self.t/self.dt)
		self.n_qubits = n_qubits
	
	def trotter_step(self,circuit,registers,k):
		H_0_temp = deepcopy(self.H_0)
		H_1_temp = deepcopy(self.H_1)
		for i in range(len(self.H_0)):
			H_0_temp[i][0] = self.H_0[i][0]*self.dt*(1 - k*self.dt/self.t)
		for i in range(len(self.H_1)):
			H_1_temp[i][0] = self.H_1[i][0]*self.dt**2*k/self.t 
		time_evolution_list = H_0_temp
		time_evolution_list.extend(H_1_temp)
		time_evolution = TimeEvolutionOperator(time_evolution_list,1,1)
		circuit,registers = time_evolution.step(circuit,registers)
		return(circuit,registers)

	def simulate(self,classical_bits):
		circuit,registers = initialize_circuit(self.n_qubits,1,classical_bits)
		circuit,registers = self.initial_state(circuit,registers)
		for k in range(self.iterations):
			circuit,registers = self.trotter_step(circuit,registers,k)
		return(circuit,registers)

	def calculate_energy(self):
		E = 0
		for h_m in self.H_1:
			factor = h_m[0]
			classical_bits = len(h_m[1:])
			if factor == 0:
				continue
			if classical_bits == 0:
				E += factor
				continue
			qubit_list = []
			circuit,registers = self.simulate(classical_bits)
			for qubit,gate in h_m[1:]:
				qubit_list.append(qubit)
				circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
			E += measure_expectation_value(qubit_list,factor,circuit,registers)
		return(E)


