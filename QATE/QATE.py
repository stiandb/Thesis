import sys
sys.path.append('../')
from utils import *
from copy import deepcopy

class QATE:
	def __init__(self,n_qubits,H_0,H_1,initial_state,dt,t=1,midpoint_method=False,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,seed_transpiler=None,transpile=False,optimization_level=1,error_mitigator=None,shots=1000):
		"""
		Input:
			n_qubits (int) - The number of qubits in the circuit
			H_0 (list)     - Hamiltonian list containing the terms for the initial hamiltonian
			H_1 (list)     - The hamiltonian to evolve to
			initial_state (function) - function accepting circuit, registers and returning these after 
			dt (float) - Time step for time evolution
			t (float) - evolution time for time evolution
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
			error_mitigator (functional) - Functional that returns the filter for error correction

		"""
		self.H_0_temp = deepcopy(H_0)
		self.H_1_temp = deepcopy(H_1)
		self.H_0 = H_0
		self.H_1 = H_1
		self.initial_state = initial_state
		self.dt = dt
		self.t = t
		self.iterations = int(self.t/self.dt)
		self.n_qubits = n_qubits
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.coupling_map=coupling_map
		self.error_mitigator = error_mitigator
		self.midpoint_method = midpoint_method
		self.h2 = None
		self.shots=shots

	
	def trotter_step(self,circuit,registers,k,steps=1,inverse=False):
		"""
		Input:
			circuit (qiskit QuantumCircuit) - the circuit to apply the adiabatic time evolution to
			registers (list) - List containing quantum registers and classical register. 
								The first register should be the register to apply the time evolution to.
								The last register should be the classical register, while the second to last register 
								should be the ancilla register to make the conditional operation required for the time evolution operation.
			k (int) - step in the adiabatic time evolution
			steps (int) - The number of steps to use in the time evolution operator

		Output:
			circuit (qiskit QuantumCircuit) - Circuit with applied adiabatic time evolution operator
			registers (list) 				- The corresponding registers
		"""
		H_0_temp = deepcopy(self.H_0)
		H_1_temp = deepcopy(self.H_1)
		for i in range(len(self.H_0)):
			if not self.midpoint_method:
				H_0_temp[i][0] = self.H_0[i][0]*self.dt*(1 - k*self.dt/self.t)
			else:
				H_0_temp[i][0] = self.H_0[i][0]*self.dt*(1 - (k*self.dt + self.dt/2)/self.t)
		for i in range(len(self.H_1)):
			if not self.midpoint_method:
				H_1_temp[i][0] = self.H_1[i][0]*self.dt*self.dt*k/self.t 
			else:
				H_1_temp[i][0] = self.H_1[i][0]*self.dt*(self.dt*k + self.dt/2)/self.t 
		time_evolution_list = H_0_temp
		time_evolution_list.extend(H_1_temp)
		time_evolution = TimeEvolutionOperator(time_evolution_list,1/steps,1,inverse=inverse)
		circuit,registers = time_evolution.step(circuit,registers)
		return(circuit,registers)

	def simulate(self,classical_bits,steps=1,early_stopping=None):
		"""
		Input:
			classical_bits (int) - The amount of classical bits / bits to measure
			steps (int) - The number of steps to use in the time evolution operator
		Output:
			circuit (qiskit QuantumCircuit) - the circuit to apply the adiabatic time evolution to
			registers (list) - List containing quantum registers and classical register. 
								The first register is the register to apply the time evolution to.
								The last register is the classical register, while the second to last register 
								is the ancilla register to make the conditional operation required for the time evolution operation.
		"""
		circuit,registers = initialize_circuit(self.n_qubits,0,classical_bits)
		circuit,registers = self.initial_state(circuit,registers)
		if early_stopping is None:
			if self.midpoint_method:
				iters = self.iterations
			else:
				iters = self.iterations+1
		else:
			iters = early_stopping
		for k in range(iters):
			circuit,registers = self.trotter_step(circuit,registers,k,steps)
		return(circuit,registers)

	def calculate_energy(self,early_stopping=None,hamiltonian_list=None):
		"""
		Output:
			E (float) - The energy for the hamiltonian we wish to solve for
		"""
		E = 0
		if not hamiltonian_list is None:
			h_list = hamiltonian_list
		else:
			h_list = self.H_1
		for h_m in h_list:
			factor = h_m[0]
			classical_bits = len(h_m[1:])
			if factor == 0:
				continue
			if classical_bits == 0:
				E += factor
				continue
			qubit_list = []
			circuit,registers = self.simulate(classical_bits,early_stopping=early_stopping)
			for qubit,gate in h_m[1:]:
				qubit_list.append(qubit)
				circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
			E += measure_expectation_value(qubit_list,factor,circuit,registers,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator,shots=self.shots)
			if not self.seed_simulator is None:
				self.seed_simulator += 1
		return(E)


