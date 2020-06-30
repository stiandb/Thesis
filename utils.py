import qiskit as qk
import numpy as np
from copy import deepcopy
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import qiskit.ignis.verification.randomized_benchmarking as rb
import collections

def QFT(circuit,registers,inverse=False):
	"""
	Inputs:
		circuit (qiskit QuantumCircuit instance)
		registers (list) - 								List containing qubit/classical registers. 
														The register to perform QFT on should be first element.
		inverse (boolean) - 							If True, the inverse QFT will be performed. 
														If left at False, a regular QFT will be performed. 
	Outputs:
		circuit (qiskit QuantumCircuit instance) - 		The inputed circuit with an applied QFT.
		registers (list) - 								List containing qubit/classical registers.
	"""
	n = len(registers[0])
	qft_circuit=qk.QuantumCircuit()
	for register in registers:
		qft_circuit.add_register(register)
	for target in range(n):
		qft_circuit.h(registers[0][target])    
		for control in range(target+1,n):
			qft_circuit.cu1(2*np.pi/(2**(control-target+1)),registers[0][control],registers[0][target])
	for qubit in range(int(n/2)):
		qft_circuit.swap(registers[0][qubit],registers[0][n-qubit-1])
	if inverse:
		qft_circuit = qft_circuit.inverse()
	circuit = circuit+qft_circuit
	return(circuit,registers)


def QPE(circuit,registers, U):
	"""
	Inputs:
		circuit (qiskit QuantumCircuit instance) - 		qiskit QuantumCircuit
		registers (list) - 								list containing the registers of circuit. The first element should be the t-register, 
														while the second should be the u-register. The final element should be the classical register.
		U (python class) - 								A class which has a function .simulate(circuit,registers,control,power). 
														The first argument is the qiskit QuantumCircuit instance.
														The second argument correpsonds to the registers argument in this function. 
														The third argument should be an int which specifies the qubit to condition on in the first register.
														The final argument is the power to put the operator to (U^(power))
														Function should return circuit and registers in the respective order.
	Outputs:
		circuit (qiskit QuantumCircuit instance) - 		A circuit with an applied QPE.
		registers (list) - 								list containing registers for circuit
	"""
	t = len(registers[0])
	for control in range(t-1,-1,-1):
		circuit.h(registers[0][control])   
		circuit, registers = U(circuit,registers,control,power=2**(t-1 - control))
	circuit, registers = QFT(circuit,registers,inverse=True)
	return(circuit,registers)



class TimeEvolutionOperatorAncilla:
	def __init__(self,hamiltonian_list,dt,T):
		self.hamiltonian_list = hamiltonian_list
		self.dt = dt
		self.T = T
		self.iters = int(self.T/self.dt)

	def step(self,circuit,registers,power=1):
		"""
		circuit (Qiskit QuantumCircuit) - 	quantum circuit to apply the time evolution operator to.
		registers (list) - 					List containing quantum registers and classical register. 
											The first register should be the register to apply the time evolution to.
											The last register should be the classical register, while the second to last register 
											should be the ancilla register to make the conditional operation required for the time evolution operation.
		power (int) - 						What power to put the operator to (U^(power))
		"""
		dt = self.dt
		for hamiltonian_term in self.hamiltonian_list:
			factor = hamiltonian_term[0]
			if factor == 0:
				continue
			qubit_and_gate = hamiltonian_term[1:]
			if len(qubit_and_gate) == 0:
				circuit.u1(-dt*power*factor, registers[0][0])
				circuit.x(registers[0][0])
				circuit.u1(-dt*power*factor, registers[0][0])
				circuit.x(registers[0][0])
				continue 
			elif len(qubit_and_gate) == 1:
				qubit= qubit_and_gate[0][0]
				gate = qubit_and_gate[0][1]
				if gate == 'x':
					circuit.rx(2*dt*factor*power,registers[0][qubit])
				elif gate == 'y':
					circuit.ry(2*dt*factor*power,registers[0][qubit])
				elif gate == 'z':
					circuit.rz(2*dt*factor*power,registers[0][qubit])
				continue

			for qubit, gate in qubit_and_gate:
				if gate == 'x':
					circuit.h(registers[0][qubit])
				elif gate == 'y':
					circuit.rz(-np.pi/2,registers[0][qubit])
					circuit.h(registers[0][qubit])
				circuit.cx(registers[0][qubit],registers[-2][0])
			circuit.rz(2*dt*factor*power,registers[-2][0])
			for qubit, gate in qubit_and_gate:
				circuit.cx(registers[0][qubit],registers[-2][0])
				if gate == 'x':
					circuit.h(registers[0][qubit])
				if gate == 'y':
					circuit.h(registers[0][qubit])
					circuit.rz(np.pi/2,registers[0][qubit])
		return(circuit,registers)

	def __call__(self,circuit,registers, power=1):
		"""
		This function is used to start the time evolution simulation
		See docstrings for the step function for variable explanation.
		"""
		for i in range(self.iters):
			circuit,registers = self.step(circuit,registers,power=power)
		return(circuit,registers)


class TimeEvolutionOperator:
	def __init__(self,hamiltonian_list,dt,T):
		self.hamiltonian_list = hamiltonian_list
		self.dt = dt
		self.T = T
		self.iters = int(self.T/self.dt)

	def step(self,circuit,registers,power=1):
		"""
		circuit (Qiskit QuantumCircuit) - 	quantum circuit to apply the time evolution operator to.
		registers (list) - 					List containing quantum registers and classical register. 
											The first register should be the register to apply the time evolution to.
											The last register should be the classical register.
		power (int) - 						What power to put the operator to (U^(power))
		"""
		dt = self.dt
		for hamiltonian_term in self.hamiltonian_list:
			factor = hamiltonian_term[0]
			if factor == 0:
				continue
			qubit_and_gate = hamiltonian_term[1:]
			if len(qubit_and_gate) == 0:
				circuit.u1(-dt*power*factor, registers[0][0])
				circuit.x(registers[0][0])
				circuit.u1(-dt*power*factor, registers[0][0])
				circuit.x(registers[0][0])
				continue 
			elif len(qubit_and_gate) == 1:
				qubit= qubit_and_gate[0][0]
				gate = qubit_and_gate[0][1]
				if gate == 'x':
					circuit.rx(2*dt*factor*power,registers[0][qubit])
				elif gate == 'y':
					circuit.ry(2*dt*factor*power,registers[0][qubit])
				elif gate == 'z':
					circuit.rz(2*dt*factor*power,registers[0][qubit])
				continue
			target = qubit_and_gate[0][0]
			for qubit, gate in qubit_and_gate:
				if gate == 'x':
					circuit.h(registers[0][qubit])
				elif gate == 'y':
					circuit.rz(-np.pi/2,registers[0][qubit])
					circuit.h(registers[0][qubit])
				if qubit != target:
					circuit.cx(registers[0][qubit],registers[0][target])
			circuit.rz(2*dt*factor*power,registers[0][target])
			qubit_and_gate.reverse()
			for qubit, gate in qubit_and_gate:
				if qubit != target:
					circuit.cx(registers[0][qubit],registers[0][target])
				if gate == 'x':
					circuit.h(registers[0][qubit])
				if gate == 'y':
					circuit.h(registers[0][qubit])
					circuit.rz(np.pi/2,registers[0][qubit])
			qubit_and_gate.reverse()
		return(circuit,registers)

	def __call__(self,circuit,registers, power=1):
		"""
		This function is used to start the time evolution simulation
		See docstrings for the step function for variable explanation.
		"""
		for i in range(self.iters):
			circuit,registers = self.step(circuit,registers,power=power)
		return(circuit,registers)

class PairingInitialState:
	def __init__(self,n_fermi):
		self.n_fermi=n_fermi
	def __call__(self,circuit,registers):
		n = len(registers[0])
		for i in range(self.n_fermi,n):
			circuit.x(registers[0][i])
		return(circuit,registers)

def pairing_initial_state(circuit,registers):
	"""
	Sets up the initial state for QPE an QITE for the pairing model
	Input:
		circuit (qiskit QuantumCircuit) - The circuit to perform the initial state ops on
		registers (list) 				- List containing registers. The first entry should be
										  the register put in initial state
	Output:
		circuit (qiskit QuantumCircuit) - Circuit with the initial state ops on
		registers (list) 				- List containing registers. The first entry is
										  the register put in initial state

	"""
	n = len(registers[0])
	for i in range(0,n,2):
		circuit.h(registers[0][i])
		circuit.cx(registers[0][i],registers[0][i+1])
	return(circuit, registers)

def pairing_ansatz(theta,circuit,registers):
	circuit.x(registers[0][2])
	circuit.x(registers[0][3])
	circuit.ry(theta[0],registers[0][0])
	circuit.cx(registers[0][0],registers[0][1])
	circuit.cx(registers[0][1],registers[0][2])
	circuit.x(registers[0][2])
	circuit.cx(registers[0][2],registers[0][3])
	circuit.x(registers[0][2])
	return(circuit,registers)


class ControlledTimeEvolutionOperator:
	"""
	This class performs the trotter approximation of the time evolution operation for an arbitrary hamiltonian.
	"""
	def __init__(self,hamiltonian_list,dt,T):
		"""
		Inputs:
			hamiltonian_list (list) - List containing the hamiltonian terms. Example: [[3,[0,'x']],[-2,[5,'y'],[3,'z']]]. 
										Each list within this list contains a term of the hamiltonian. 
										The first term has a factor of 3 and we apply a pauli-x gate to the first qubit (qubit index 0). 
										The second term has a factor -2 and we apply a pauli-y gate to the 6th qubit (qubit index 5) and a 
										pauli-z gate to the 4th qubit (qubit index 3).
			dt (float) -  			The time step applied with the time evolution operator	
			T (float) -				The time to apply the operator	
		
		"""
		self.hamiltonian_list = hamiltonian_list
		self.dt = dt
		self.T = T
		self.iters = int(T/dt)
	def step(self,circuit,registers,control,power=1):
		"""
		circuit (Qiskit QuantumCircuit) - 	quantum circuit to apply the time evolution operator to.
		registers (list) - 					List containing quantum registers and classical register. 
											The first register should be the t-register for the phase estimation algorithm. 
											The second register should be the register to apply the time evolution to.
											The last register should be the classical register, while the second to last register 
											should be the ancilla register to make the conditional operation required for the time evolution operation.
		control (Int) - 					The control qubit for the controlled time evolution operation.
											should be the ancilla register to make the conditional operation required for the time evolution operation.
		power (int) - 						What power to put the operator to (U^(power))
		"""
		dt = self.dt
		for hamiltonian_term in self.hamiltonian_list:
			factor = hamiltonian_term[0]
			if factor == 0:
				continue
			qubit_and_gate = hamiltonian_term[1:]
			if len(qubit_and_gate) == 0:
				circuit.cu1(-dt*power*factor, registers[0][control],registers[1][0])
				circuit.x(registers[1][0])
				circuit.cu1(-dt*power*factor, registers[0][control],registers[1][0])
				circuit.x(registers[1][0])
				continue 
			elif len(qubit_and_gate) == 1:
				qubit= qubit_and_gate[0][0]
				gate = qubit_and_gate[0][1]
				if gate == 'x':
					circuit.mcrx(2*dt*factor*power,[registers[0][control]], registers[1][qubit])
				elif gate == 'y':
					circuit.cry(2*dt*factor*power,registers[0][control], registers[1][qubit])
				elif gate == 'z':
					circuit.crz(2*dt*factor*power,registers[0][control], registers[1][qubit])
				continue
			target = qubit_and_gate[0][0]
			for qubit, gate in qubit_and_gate:
				if gate == 'x':
					circuit.ch(registers[0][control],registers[1][qubit])
				elif gate == 'y':
					circuit.crz(-np.pi/2,registers[0][control],registers[1][qubit])
					circuit.ch(registers[0][control],registers[1][qubit])
				if qubit != target:
					circuit.ccx(registers[0][control],registers[1][qubit],registers[1][target])
			circuit.crz(2*dt*factor*power,registers[0][control],registers[1][target])
			qubit_and_gate.reverse()
			for qubit, gate in qubit_and_gate:
				if qubit != target:
					circuit.ccx(registers[0][control],registers[1][qubit],registers[1][target])
				if gate == 'x':
					circuit.ch(registers[0][control],registers[1][qubit])
				if gate == 'y':
					circuit.ch(registers[0][control],registers[1][qubit])
					circuit.crz(np.pi/2,registers[0][control],registers[1][qubit])
			qubit_and_gate.reverse()
		return(circuit,registers)
	def __call__(self,circuit,registers,control, power=1):
		"""
		This function is used to start the time evolution simulation
		See docstrings for the step function for variable explanation.
		"""
		for i in range(self.iters):
			circuit,registers = self.step(circuit,registers,control=control,power=power)
		return(circuit,registers)


def pairing_hamiltonian(n_states,delta,g):
	"""
	This function gives out the pairing hamiltonian in a format recognized by the 
	methods developed in this thesis. 
	Inputs:
		n_states (int) - The number of basis states / qubits used in the model.
		delta (float) - One body interaction term
		g (float) - Two body interaction term
	Output:
		hamiltonian_list (list) - List containing each term of the hamiltonian in terms of factors, qubit number and gates.
	"""
	hamiltonian_list = []
	H_0 = []
	V = []
	phase_H_0 = 0
	phase_V = 0
	for p in range(0,int(n_states)):
		if (p+1 - 1 - (1 if (p+1)%2 == 0 else 0)) != 0 and delta != 0:
			phase_H_0 += 0.5*delta*0.5*(p+1 - 1 - (1 if (p+1)%2 == 0 else 0))
			H_0.append([0.5*delta*0.5*(p+1 - 1 - (1 if (p+1)%2 == 0 else 0)),[p,'z']])
		if g != 0 and p < int(n_states/2):
			phase_V += -(1/8)*g
			V.append([-(1/8)*g,[2*p,'z']])
			V.append([-(1/8)*g,[2*p+1,'z']])
			V.append([-(1/8)*g,[2*p,'z'],[2*p+1,'z']])
			for q in range(p+1,int(n_states/2)):
				V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'x'],[2*q,'x'],[2*q+1,'x']])
				V.append([(1/16)*g,[2*p,'x'],[2*p+1,'x'],[2*q,'y'],[2*q+1,'y']])
				V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'y'],[2*q,'x'],[2*q+1,'y']])
				V.append([-(1/16)*g,[2*p,'x'],[2*p+1,'y'],[2*q,'y'],[2*q+1,'x']])
				V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'x'],[2*q,'x'],[2*q+1,'y']])
				V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'x'],[2*q,'y'],[2*q+1,'x']])
				V.append([(1/16)*g,[2*p,'y'],[2*p+1,'y'],[2*q,'x'],[2*q+1,'x']])
				V.append([-(1/16)*g,[2*p,'y'],[2*p+1,'y'],[2*q,'y'],[2*q+1,'y']])
	hamiltonian_list = H_0
	hamiltonian_list.extend(V)
	hamiltonian_list.append([phase_H_0+phase_V])
	return(hamiltonian_list)

def pauli_expectation_transformation(qubit,gate,circuit,registers):
		"""
		Applies a gate and its corresponding transformation to obtain eigenvalues
		"""
		if gate == 'x':
			circuit.x(registers[0][qubit])
			circuit.h(registers[0][qubit])
		elif gate == 'y':
			circuit.y(registers[0][qubit])
			circuit.sdg(registers[0][qubit])
			circuit.h(registers[0][qubit])
		elif gate == 'z':
			circuit.z(registers[0][qubit])
		return(circuit,registers)

class ErrorMitigation:
	def __init__(self):
		self.filters = {}

	def __call__(self,n_qubits,qubit_list,backend,seed_simulator=None,noise_model=None,basis_gates=None,coupling_map=None,shots=1000,transpile=False,seed_transpiler=None,optimization_level=1,initial_layout=None):
		try:
			self.filters[str(qubit_list).strip('[]')]
		except:
			self.filters[str(qubit_list).strip('[]')] = self.get_meas_fitter(n_qubits,qubit_list,backend,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,shots=1000,transpile=transpile,seed_transpiler=seed_transpiler,optimization_level=optimization_level,initial_layout=initial_layout).filter
		meas_filter=self.filters[str(qubit_list).strip('[]')]
		return(meas_filter)

	def get_meas_fitter(self,n_qubits,qubit_list,backend,seed_simulator=None,noise_model=None,basis_gates=None,coupling_map=None,shots=1000,transpile=False,seed_transpiler=None,optimization_level=1,initial_layout=None):
		register = qk.QuantumRegister(n_qubits)
		meas_calibs, state_labels = complete_meas_cal(qr=register,circlabel='mcal')#complete_meas_cal(qubit_list,qr=register,circlabel='mcal')
		if transpile:
			meas_calibs = qk.compiler.transpile(meas_calibs,backend=backend,backend_properties=backend.properties(),seed_transpiler=seed_transpiler,optimization_level=optimization_level,basis_gates=basis_gates,coupling_map=coupling_map)
			layout = meas_calibs[0]._layout.get_virtual_bits()
			qubit_list = [list(layout.values())[key] for key in qubit_list]

		job = qk.execute(meas_calibs,
					  backend=backend,
					  shots=shots,
					  noise_model=noise_model,
					  coupling_map=coupling_map,
					  basis_gates=basis_gates,
					  seed_simulator=seed_simulator
					  )
		calibration_results = job.result()
		meas_fitter = CompleteMeasFitter(calibration_results, state_labels, circlabel='mcal')
		meas_fitter = meas_fitter.subset_fitter(qubit_list)
		return(meas_fitter)


def measure_expectation_value(qubit_list,factor,circuit,registers,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,shots=1000,transpile=False,seed_transpiler=None,optimization_level=1,coupling_map=None,error_mitigator=None,initial_layout=None):
	"""
	Measures qubits and calculates eigenvalues
	"""
	circuit.measure([registers[0][qubit] for qubit in qubit_list],registers[-1])
	if transpile:
		circuit = qk.compiler.transpile(circuit,backend=backend,backend_properties=backend.properties(),seed_transpiler=seed_transpiler,optimization_level=optimization_level,basis_gates=basis_gates,coupling_map=coupling_map)
		initial_layout = circuit._layout.get_virtual_bits()
	job = qk.execute(circuit, backend = backend,seed_simulator=seed_simulator,shots=shots,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map).result()
	if not error_mitigator is None:
		try:
			n_qubits = circuit.n_qubits
		except:
			n_qubits = circuit.num_qubits
		meas_filter = error_mitigator(n_qubits,qubit_list,backend,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,shots=shots,transpile=transpile,seed_transpiler=seed_transpiler,optimization_level=optimization_level,initial_layout=initial_layout)
		result = meas_filter.apply(job)
		result = result.get_counts(0)
	else:
		result = job.get_counts(circuit)
	E = 0
	for key, value in result.items():
		key1 = key[::-1]
		eigenval = 1
		for bit in key1:
			e =  1 if bit == '0' else -1
			eigenval *= e
		E += eigenval*value
	E /= shots
	return(factor*E)


def initialize_circuit(n_qubits,n_ancilla,classical_bits):
	"""
	Initializes circuits for experiment
	"""
	simulation_register = qk.QuantumRegister(n_qubits)
	classical_register = qk.ClassicalRegister(classical_bits)
	if n_ancilla != 0:
		ancilla_register = qk.QuantumRegister(n_ancilla)
		circuit = qk.QuantumCircuit(simulation_register,ancilla_register,classical_register)
		return(circuit,[simulation_register,ancilla_register,classical_register])
	else:
		circuit = qk.QuantumCircuit(simulation_register,classical_register)
		return(circuit,[simulation_register,classical_register])

def max_cut_hamiltonian(W):
	"""
	This function returns the hamiltonian list for any max cut matrix W
	Input:
		W (numpy 2D array) - Max cut matrix
	Output:
		hamiltonian_list (list) - hamiltonian_list for the given max cut problem
	"""
	hamiltonian_list= []
	for i in range(W.shape[0]):
		for j in range(i+1, W.shape[0]):
			hamiltonian_list.append([W[i,j],[i,'z'],[j,'z']])
	return(hamiltonian_list)


def y_rotation_ansatz(theta,circuit,registers):
	"""
	Applies the R_y rotation ansatz to a quantum circuit
	Input:
		theta (numpy array) - 1D array containing all variational parameters
		circuit (qiskit quantum circuit instance) - quantum circuit
		registers (list) - list containing the ansatz register as first element
	Output:
		circuit (qiskit quantum circuit instance) - circuit with applied ansatz
		registers (list) - the corresponding list
	"""
	if theta.shape[0] != len(registers[0]):
		print('y_rotation_ansatz warning: Number of parameters does not match the number of qubits')
	for qubit,param in enumerate(theta):
		circuit.ry(param,registers[0][qubit])
	for i in range(theta.shape[0]-1):
		circuit.cx(registers[0][i],registers[0][i+1])
	return(circuit,registers)

class EulerRotationAnsatz:
	"""Euler rotation ansatz with depth d and n qubits. Requires 3dn parameters"""
	def __init__(self,entangler):
		"""
		Input:
			entangler (callable) - accepts parameters (circuit,registers) and entangles the first register
		"""
		self.entangler = entangler
	def __call__(self,theta,circuit,registers):
		"""
		Input:
			theta (numpy array) - 1D array, which should contain 3nd elements
			circuit (qiskit quantum circuit instance) - quantum circuit
			registers (list) - list containing the ansatz register as first element
		Output:
			circuit (qiskit quantum circuit instance) - circuit with applied ansatz
			registers (list) - the corresponding list
		"""
		n = len(registers[0])
		D = int(theta.shape[0]/(3*n))
		i = 0
		for d in range(D):
			for q in range(n):
				circuit.rz(theta[i],registers[0][q])
				circuit.rx(theta[i+1],registers[0][q])
				circuit.rz(theta[i+2],registers[0][q])
				i+=3
			circuit,registers = self.entangler(circuit,registers)
		return(circuit,registers)

def identity_circuit(circuit,registers):
	return(circuit,registers)

def identity_ansatz(theta,circuit,registers):
	return(circuit,registers)

def linear_entangler(circuit,registers):
	"""
	Entangles qubit i with qubit i+1 for all n qubits in the first element of registers
	Input:
		circuit (qiskit quantum circuit instance) - quantum circuit
		registers (list) - list containing the ansatz register as first element
	Output:
		circuit (qiskit quantum circuit instance) - quantum circuit
		registers (list) - list containing the ansatz register as first element
	"""
	n = len(registers[0])
	for j in range(n-1):
		circuit.cx(registers[0][j],registers[0][j+1])
	return(circuit,registers)


class UCCSD:
	def __init__(self,n_fermi,n_spin_orbitals,initial_state,t,dt=1,T=1,singles=True,doubles=True):
		self.n_fermi = n_fermi
		self.n_spin_orbitals = n_spin_orbitals
		self.singles = singles
		self.doubles = doubles
		self.hamiltonian_list = []
		self.dt = dt
		self.T = T
		self.initial_state = initial_state
		self.create_hamiltonian_list(t)


	def __call__(self,t,circuit,registers):
		singles = self.singles
		doubles = self.doubles
		if singles and not doubles:
			self.replace_parameters_singles(t)
		elif doubles and not singles:
			self.replace_parameters_doubles(t)
		else:
			theta = t[:self.n_fermi*(self.n_spin_orbitals-self.n_fermi)]
			self.replace_parameters_singles(theta)
			theta = t[self.n_fermi*(self.n_spin_orbitals - self.n_fermi):]
			self.replace_parameters_doubles(theta)
		time_evolution = TimeEvolutionOperator(self.hamiltonian_list,self.dt,self.T)
		circuit,registers = self.initial_state(circuit,registers)
		circuit,registers = time_evolution(circuit,registers)
		return(circuit,registers)

	def singles_operator(self,t):
		n_fermi = self.n_fermi
		n_spin_orbitals = self.n_spin_orbitals
		idx=0
		for i in range(n_fermi):
			for a in range(n_fermi,n_spin_orbitals):
				z_gates = []
				for k in range(i+1,a):
					z_gates.append([k,'z'])
				term_1 = [t[idx]/2,[i,'y'],[a,'x']]
				term_1.extend(z_gates)
				term_2 = [-t[idx]/2,[i,'x'],[a,'y']]
				term_2.extend(z_gates)
				idx += 1
				self.hamiltonian_list.append(term_1)
				self.hamiltonian_list.append(term_2)
	
	def doubles_operator(self,t):
		n_fermi = self.n_fermi
		n_spin_orbitals = self.n_spin_orbitals
		idx = 0
		for i in range(n_fermi):
			for j in range(i+1,n_fermi):
				for a in range(n_fermi,n_spin_orbitals):
					for b in range(a+1,n_spin_orbitals):
						z_gates = []
						for k in range(i+1,j):
							z_gates.append([k,'z'])
						for l in range(a+1,b):
							z_gates.append([l,'z'])
						term_1 = [t[idx]/8,[i,'x'],[j,'x'],[a,'y'],[b,'x']]
						term_1.extend(z_gates)
						term_2 = [t[idx]/8,[i,'y'],[j,'x'],[a,'y'],[b,'y']]
						term_2.extend(z_gates)
						term_3 = [t[idx]/8,[i,'x'],[j,'y'],[a,'y'],[b,'y']]
						term_3.extend(z_gates)
						term_4 = [t[idx]/8,[i,'x'],[j,'x'],[a,'x'],[b,'y']]
						term_4.extend(z_gates)
						term_5 = [-t[idx]/8,[i,'y'],[j,'x'],[a,'x'],[b,'x']]
						term_5.extend(z_gates)
						term_6 = [-t[idx]/8,[i,'x'],[j,'y'],[a,'x'],[b,'x']]
						term_6.extend(z_gates)
						term_7 = [-t[idx]/8,[i,'y'],[j,'y'],[a,'y'],[b,'x']]
						term_7.extend(z_gates)
						term_8 = [-t[idx]/8,[i,'y'],[j,'y'],[a,'x'],[b,'y']]
						term_8.extend(z_gates)
						self.hamiltonian_list.append(term_1)
						self.hamiltonian_list.append(term_2)
						self.hamiltonian_list.append(term_3)
						self.hamiltonian_list.append(term_4)
						self.hamiltonian_list.append(term_5)
						self.hamiltonian_list.append(term_6)
						self.hamiltonian_list.append(term_7)
						self.hamiltonian_list.append(term_8)
						idx+=1

	def create_hamiltonian_list(self,t):
		singles = self.singles
		doubles = self.doubles
		if singles:
			theta = t[:self.n_fermi*(self.n_spin_orbitals - self.n_fermi)]
			self.singles_operator(theta)
		if doubles and singles:
			theta = t[self.n_fermi*(self.n_spin_orbitals - self.n_fermi):]
			self.doubles_operator(theta)
		if doubles and not singles:
			self.doubles_operator(t)
		return(self.hamiltonian_list)

	def replace_parameters_singles(self,t):
		idx = 0
		for i in range(self.n_fermi):
			for a in range(self.n_fermi,self.n_spin_orbitals):
				self.hamiltonian_list[idx][0] = t[int(idx/2)]/2
				self.hamiltonian_list[idx+1][0] = -t[int(idx/2)]/2
				idx += 2

	def replace_parameters_doubles(self,t):
		singles = self.singles
		if singles:
			idx = self.n_fermi*(self.n_spin_orbitals-self.n_fermi)
		else:
			idx = 0
		for i in range(self.n_fermi):
			for j in range(i+1,self.n_fermi):
				for a in range(self.n_fermi,self.n_spin_orbitals):
					for b in range(a+1,self.n_spin_orbitals):
						self.hamiltonian_list[idx][0] = t[int(idx/8)]/8
						self.hamiltonian_list[idx+1][0] = t[int(idx/8)]/8
						self.hamiltonian_list[idx+2][0] = t[int(idx/8)]/8
						self.hamiltonian_list[idx+3][0] = t[int(idx/8)]/8
						self.hamiltonian_list[idx+4][0] = -t[int(idx/8)]/8
						self.hamiltonian_list[idx+5][0] = -t[int(idx/8)]/8
						self.hamiltonian_list[idx+6][0] = -t[int(idx/8)]/8
						self.hamiltonian_list[idx+7][0] = -t[int(idx/8)]/8
						idx += 8





class PairingUCCD:
	def __init__(self,n_fermi,n_spin_orbitals,initial_state,t,dt=1,T=1):
		"""
		Input:
			n_fermi (int) - The number of particles in the system
			n_spin_orbitals (int) - The number of spin orbitals in the system
			t (numpy array) - 1D array with the UCCD amplitudes
			dt (float) - Time step in the trotter approximation
			T (float) - Total time in the trotter approximation
		"""
		self.n_fermi = n_fermi
		self.n_spin_orbitals = n_spin_orbitals
		self.hamiltonian_list = []
		self.dt = dt
		self.T = T
		self.initial_state=initial_state
		self.doubles_operator(t)


	def __call__(self,t,circuit,registers):
		"""
		Input:
			t (numpy array) - 1D array with the UCCD amplitudes
			circuit (qiskit QuantumCircuit) - The circuit to apply the UCCD ansatz to
			registers (list) - List containing qiskit registers. The first register should be the one to apply the ansatz to.
								the second to last should be an ancilla register
								the final should be the classical register
		Outputs:
			circuit (qiskit QuantumCircuit) - The circuit with an applied UCCD ansatz
			registers (list) - List containing the corresponding registers
		"""
		n_fermi = self.n_fermi
		n_spin_orbitals = self.n_spin_orbitals
		self.replace_parameters(t)
		time_evolution = TimeEvolutionOperator(self.hamiltonian_list,self.dt,self.T)
		circuit,registers=self.initial_state(circuit,registers)
		circuit,registers = time_evolution(circuit,registers)
		return(circuit,registers)

	
	def doubles_operator(self,t):
		"""
		Calculates the doubles operator for arbitrary amplitudes t
		"""
		n_fermi = self.n_fermi
		n_spin_orbitals = self.n_spin_orbitals
		idx=0
		for i in range(0,n_fermi,2):
			for a in range(n_fermi,n_spin_orbitals,2):
				self.hamiltonian_list.append([t[idx]/8,[i,'x'],[i+1,'x'],[a,'y'],[a+1,'x']])
				self.hamiltonian_list.append([t[idx]/8,[i,'y'],[i+1,'x'],[a,'y'],[a+1,'y']])
				self.hamiltonian_list.append([t[idx]/8,[i,'x'],[i+1,'y'],[a,'y'],[a+1,'y']])
				self.hamiltonian_list.append([t[idx]/8,[i,'x'],[i+1,'x'],[a,'x'],[a+1,'y']])
				self.hamiltonian_list.append([-t[idx]/8,[i,'y'],[i+1,'x'],[a,'x'],[a+1,'x']])
				self.hamiltonian_list.append([-t[idx]/8,[i,'x'],[i+1,'y'],[a,'x'],[a+1,'x']])
				self.hamiltonian_list.append([-t[idx]/8,[i,'y'],[i+1,'y'],[a,'y'],[a+1,'x']])
				self.hamiltonian_list.append([-t[idx]/8,[i,'y'],[i+1,'y'],[a,'x'],[a+1,'y']])
				idx+=1


	def replace_parameters(self,t):
		"""
		Replaces the amplitudes in hamiltonian_list with new amplitudes t
		"""
		idx = 0
		for i in range(0,self.n_fermi,2):
			for a in range(self.n_fermi,self.n_spin_orbitals,2):
				
				self.hamiltonian_list[idx][0] = t[int(idx/8)]/8
				self.hamiltonian_list[idx+1][0] = t[int(idx/8)]/8
				self.hamiltonian_list[idx+2][0] = t[int(idx/8)]/8
				self.hamiltonian_list[idx+3][0] = t[int(idx/8)]/8
				self.hamiltonian_list[idx+4][0] = -t[int(idx/8)]/8
				self.hamiltonian_list[idx+5][0] = -t[int(idx/8)]/8
				self.hamiltonian_list[idx+6][0] = -t[int(idx/8)]/8
				self.hamiltonian_list[idx+7][0] = -t[int(idx/8)]/8
				idx += 8



class PairingSimpleUCCDAnsatz:
	def __init__(self,initial_state,dt=1,T=1):
		self.n_fermi = 2
		self.n_spin_orbitals = 4
		self.hamiltonian_list = []
		self.dt = dt
		self.T = T
		self.initial_state=initial_state


	def __call__(self,theta,circuit,registers):
		i=0
		j=i+1
		a=2
		b=a+1
		term = [[theta[0],[i,'y'],[j,'x'],[a,'x'],[b,'x']]]
		time_evolution = TimeEvolutionOperator(term,self.dt,self.T)
		circuit,registers=self.initial_state(circuit,registers)
		circuit,registers = time_evolution(circuit,registers)
		return(circuit,registers)

	


				

def hadamard_test(circuit,registers,hamiltonian_term,imag=True,shots=1000,ancilla_index=-2,backend=qk.Aer.get_backend('qasm_simulator'),seed_simulator=None,noise_model=None,basis_gates=None,transpile=False,seed_transpiler=None,optimization_level=1,coupling_map=None,error_mitigator=None):
	"""
	Input:
		circuit (qiskit QuantumCircuit) - Circuit to apply Hadamard test on.
		registers (list) - List containing qiskit registers. The first register
							is the one to apply the operator on.
							The second to last register is the ancilla register for the test.
							The last register is the classical register to measure.
		hamiltonian_term (class instance) - class instance with call method that accepts circuit and registers and returns
											circuit,registers with the hamiltonian term applied. hamiltonian_term.factor
											should output the factor for the term
		imag (boolean) - If True (default), calculate the imaginary part. If false, calculate the real part
		shots (int)    - How many measurements to make on the ancilla.
	Output:
		measurements (float) - The real or imaginary part.
	"""
	circuit.h(registers[ancilla_index][0])
	if imag:
		circuit.sdg(registers[ancilla_index][0])
	if type(hamiltonian_term) == list:
		factor = hamiltonian_term[0]
		for qubit_and_gate in hamiltonian_term[1:]:
			qubit = qubit_and_gate[0]
			gate = qubit_and_gate[1]
			if gate == 'x':
				circuit.x(registers[0][qubit])
			elif gate == 'y':
				circuit.y(registers[0][qubit])
			elif gate == 'z':
				circuit.z(registers[0][qubit])
	else:
		circuit,register = hamiltonian_term(circuit,registers)
		factor = hamiltonian_term.factor
	circuit.h(registers[ancilla_index][0])
	circuit.measure(registers[ancilla_index][0],registers[-1])
	if transpile:
		circuit = qk.compiler.transpile(circuit,backend=backend,backend_properties=backend.properties(),seed_transpiler=seed_transpiler,optimization_level=optimization_level,basis_gates=basis_gates,coupling_map=coupling_map)
	job = qk.execute(circuit, backend = backend, shots=shots,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map).result()
	if not error_mitigator is None:
		meas_filter = error_mitigator(circuit.num_qubits,[-1],backend,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,shots=shots)
		result = meas_filter.apply(job)
		result = result.get_counts(circuit)
	else:
		result = job.get_counts(circuit)
	measurements = 0
	for key,value in result.items():
		if key == '0':
			measurements += value/shots
		if key == '1':
			measurements -= value/shots
	return(measurements*factor)


def operator_product(operator_1, operator_2):
	"""
	Calculates the product of two one-termed operators containing pauli-gates.
	Input:
		operator_1 (list) - Example: [5,[0,'z'],[3,y]] is a two-qubit operator which acts on the first qubit
							with the pauli-z gate and the fourth qubit with the pauli-y gate with a factor of five.
		operator_2 (list) - Example: [5,[0,'z'],[3,y]] is a two-qubit operator which acts on the first qubit
							with the pauli-z gate and the fourth qubit with the pauli-y gate with a factor of five.
	Output:
		operator (list) -  The product of the two operator in the arguments
	"""
	
	
	if len(operator_1) == 1:
		operator = deepcopy(operator_2)
		operator[0] *= operator_1[0]
		return(operator)
	if len(operator_2) == 1:
		operator = deepcopy(operator_1)
		operator[0] *= operator_2[0]
		return(operator)
	operator = deepcopy(operator_2)
	factor=operator_1[0]*operator_2[0]
	qubit_list = []
	for qubit_1, gate_1 in operator_1[1:]:
		qubit_match = False
		idx = 0
		for qubit_2,gate_2 in operator_2[1:]:
			idx += 1
			if qubit_1 == qubit_2:
				qubit_match = True
				if gate_1 == 'x' and gate_2 == 'y':
					factor *= np.complex(0,1)
					operator[idx] = [qubit_1,'z']
				elif gate_1 == 'y' and gate_2 == 'x':
					factor *= -np.complex(0,1)
					operator[idx] = [qubit_1,'z']
				elif gate_1 == 'x' and gate_2 == 'z':
					factor *= -np.complex(0,1)
					operator[idx] = [qubit_1,'y']
				elif gate_1 == 'z' and gate_2 == 'x':
					factor *= np.complex(0,1)
					operator[idx] = [qubit_1,'z']
				elif gate_1 == 'y' and gate_2 == 'z':
					factor *= np.complex(0,1)
					operator[idx] = [qubit_1,'x']
				elif gate_1 == 'z' and gate_2 == 'y':
					factor *= -np.complex(0,1)
					operator[idx] = [qubit_1,'x']
				elif gate_1 == gate_2:
					qubit_list.append(qubit_1)
				break
		if not qubit_match:
			operator.append([qubit_1,gate_1])
	if len(qubit_list) != 0:
		for qubit,gate in operator[1:]:
			if qubit in qubit_list:
				operator.remove([qubit,gate])
			idx += 1
	operator[0] = factor
	return(operator)

class AmplitudeEncoder:
	"""
	Encodes an arbitrary vector into the amplitudes of a quantum state
	"""
	def __init__(self,eps = 1e-14):
		"""
		Input:
			eps (float) - In case of zero vectors, this is used to prevent dividing by zero when normalizing the dataset
		"""
		self.eps=eps
		self.n_qubits = None
		self.usage = None

	def make_amplitude_list(self,X):
		"""
		Input:
			X (numpy array) - Array containing dataset to encode into the quantum state
		"""
		if X.flatten().shape[0] == 1:
			X_ = np.zeros(2)
			X_[0] = X[0]
			X_[1] = 0.5
			X = X_
		X = X/(np.sqrt(np.sum(X**2))+self.eps)
		values = list(X.copy().flatten())
		if self.n_qubits is None:
			self.n_qubits = int(np.ceil(np.log2(len(values))))
		else:
			self.n_qubits
		n_ancilla = 1 if (self.n_qubits - 2 <= 1) else (self.n_qubits - 2)
		ancilla_register = qk.QuantumRegister(n_ancilla)
		keys = self.convert_binary(range(len(values)))
		if keys[-1][:-1] != keys[-2][:-1]:
			keys.append(keys[-1][:-1] + '1')
			values.append(0)
		amplitude_list = [[k, v] for k,v in zip(keys,values)]
		return(amplitude_list,ancilla_register)

	def convert_binary(self,idx):
		"""
		Converts a list containing indexes to binary numbers.
		Input:
			idx (list) - List containing indexes
		Output:
			binary_list (list) - List containing the corresponding binary numbers
		"""
		get_bin = lambda x: format(x,'b')
		binary_list = []
		for i in idx:
			binary = get_bin(i)
			if len(binary) !=  self.n_qubits:
				for j in range(self.n_qubits - len(binary)):
					 binary = '0' + binary
			binary_list.append(binary)
		return(binary_list)

	def iterate(self,amplitude_list,circuit,registers):
		"""
		Performs a step in the amplitude encoding algorithm. That is, utilizes y-rotation to zero out one qubit

		Input:
			amplitude_list (list) - List returned by the make_amplitude_list function
			circuit (qiskit QuantumCircuit) - the circuit to perform the amplitude encoding on
			registers (list) - List containing qiskit QuantumRegisters. The first should be the register to appy the encoding to
								The final should be the classical register
		Output:
			new_amplitude_list (list) - List containing the amplitudes and binary numbers for the next step
			circuit (qiskit QuantumCircuit) - the circuit to perform the amplitude encoding on
			registers (list) - List containing qiskit QuantumRegisters. The first should be the register to appy the encoding to
								The final should be the classical register
		"""
		new_amplitudes = []
		for idx in range(len(amplitude_list)-1,-1,-2):
			bit_string, amp1 = amplitude_list[idx]
			amp0 = amplitude_list[idx-1][1]
			if amp0 == 0 and amp1 == 0:
				new_amplitudes.insert(0,[bit_string[:-1],0])
				continue
			elif amp1 == 0:
				theta = 0
				amp = amp0
				new_amplitudes.insert(0,[bit_string[:-1],amp])
				continue
			elif amp0 == 0:
				theta = -np.pi
				amp = amp1
			else:
				theta = -2*np.arctan(amp1/amp0)
				amp = amp0*np.sqrt(amp1**2/amp0**2 + 1)
			for qbit,bit in enumerate(bit_string[:-1]):
				if bit == '0':
					circuit.x(registers[0][qbit])
			control_qubits = [registers[0][i] for i in range(len(bit_string[:-1]))]
			circuit.mcry(theta, control_qubits, registers[0][(len(bit_string)-1)], registers[-2])
			new_amplitudes.insert(0,[bit_string[:-1],amp])
			for qbit,bit in enumerate(bit_string[:-1]):
				if bit == '0':
					circuit.x(registers[0][qbit])
		return(new_amplitudes,circuit,registers)

	def __call__(self,circuit,registers,X,inverse=False):
		"""
		Amplitude encodes a vector X
		Input:
			circuit (qiskit QuantumCircuit) - the circuit to perform the amplitude encoding on
			registers (list) - List containing qiskit QuantumRegisters. The first should be the register to appy the encoding to
								The final should be the classical register
			X (numpy array) - The vector to amplitude encode into the quantum state
			inverse (boolean) - If put to True, the inverse amplitude encoding circuit will be added to the inputed circuit
		Output:
			circuit (qiskit QuantumCircuit) - the circuit to perform the amplitude encoding on
			registers (list) - List containing qiskit QuantumRegisters. 
								The first is the register with an applied encoding
								The final is the classical register
								The next to final register is added by this class and is the ancilla register 
								used for the multi-controlled gates
		"""
		amplitude_list, ancilla_register = self.make_amplitude_list(X)
		n = len(registers[0])
		encoded_circuit=qk.QuantumCircuit()
		for register in registers:
			encoded_circuit.add_register(register)
		if self.usage is None:
			encoded_circuit.add_register(ancilla_register)
			circuit.add_register(ancilla_register)
			registers.insert(-1,ancilla_register)
		if np.all(X == X[0]):
			for i in range(n):
				encoded_circuit.h(registers[0][i])
			circuit += encoded_circuit
			self.usage = True
			return(circuit,registers)
		for i in range(self.n_qubits):
			if len(amplitude_list) > 2:
				amplitude_list,encoded_circuit,registers = self.iterate(amplitude_list,encoded_circuit,registers)
			else:
				amp1 = amplitude_list[1][1]
				amp0 = amplitude_list[0][1]
				if amp0 == 0 and amp1 == 0:
					continue
				elif amp1 == 0:
					encoded_circuit.ry(0,registers[0][0])
				elif amp0 == 0:
					encoded_circuit.ry(-np.pi,registers[0][0])
				if amp0 != 0:
					encoded_circuit.ry(-2*np.arctan(amp1/amp0),registers[0][0])
		if not inverse:
			encoded_circuit = encoded_circuit.inverse()
		circuit += encoded_circuit
		self.usage = True
		return(circuit,registers)			
				
def squared_inner_product(x,y,circuit,registers,shots=1000,backend=qk.Aer.get_backend('qasm_simulator'),seed_simulator=None,noise_model=None,basis_gates=None,transpile=False,seed_transpiler=None,optimization_level=1,coupling_map=None,error_mitigator=None):
	"""
	Calculates the squared inner product of two arbitrary vectors.
	Input:
		x (numpy array) - The first vector
		y (numpy array) - The second vector
		circuit (qiskit QuantumCircuit) - The circuit to calculate the inner product with
		registers (list) - List containing the amplitude register as first element, while the last
							element should be a classical register with one bit
		shots (int) - How many times to measure the circuit when calculating the squared inner product
	Output:
		inner_product (float) - An estimate of the squared inner product of x and y.
	"""
	encoder= AmplitudeEncoder()
	ancilla_register = qk.QuantumRegister(1)
	circuit.add_register(ancilla_register)
	circuit,registers = encoder(circuit,registers,x)
	circuit,registers = encoder(circuit,registers,y,inverse=True)
	for i in range(len(registers[0])):
		circuit.x(registers[0][i])
	circuit.mcrx(np.pi,[registers[0][i] for i in range(len(registers[0]))],ancilla_register[0])
	circuit.measure(ancilla_register,registers[-1])
	if transpile:
		circuit = qk.compiler.transpile(circuit,backend=backend,backend_properties=backend.properties(),seed_transpiler=seed_transpiler,optimization_level=optimization_level,basis_gates=basis_gates,coupling_map=coupling_map)
	job = qk.execute(circuit, backend = backend, shots=shots,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map).result()
	if not error_mitigator is None:
		meas_filter = error_mitigator(circuit.num_qubits,[-1],backend,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,shots=shots)
		result = meas_filter.apply(job)
		result = result.get_counts(circuit)
	else:
		result = job.get_counts(circuit)
	inner_product = 0
	for key,value in result.items():
		if key == '1':
			inner_product += value
	inner_product /= shots
	return(inner_product)











