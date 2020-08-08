import sys
sys.path.append('../')
from utils import *

class HamiltonianSimulation:
	def __init__(self,u_qubits,t_qubits,hamiltonian_list,initial_state,shots=1000,backend=qk.Aer.get_backend('qasm_simulator'),seed_simulator=None,noise_model=None,basis_gates=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		"""
		Input:
			u_qubits (int) - the number of qubits to utilize for the u-register in the QPE algorithm
			t_qubits (int) - the number of qubits to utilize for the t-register in the QPE algorithm
			hamiltonian_list (list) - List containing the hamiltonian terms for the QPE algorthm
			initial_state (functional) - initial_state(circuit,registers) returns circuit,registers in some initial state
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
			error_mitigator (functional) - a function that returns the filter for error correction
		"""
		self.hamiltonian_list = hamiltonian_list
		self.u_qubits = u_qubits
		self.t_qubits = t_qubits
		self.t_register = qk.QuantumRegister(t_qubits)
		self.circuit,self.registers= initialize_circuit(u_qubits,0,t_qubits)
		self.circuit.add_register(self.t_register)
		self.initial_state = initial_state
		self.shots=shots
		self.backend = backend
		self.seed_simulator=seed_simulator
		self.noise_model=noise_model
		self.basis_gates=basis_gates
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.coupling_map = coupling_map
		self.error_mitigator = error_mitigator

	def __call__(self,dt,t):
		"""
		Input:
			dt (float) - The time step of the time evolution operation
			t (float) - The evolution time for the time evolution oepration
		Output:
			circuit,registers:
				circuit (qiskit QuantumCircuit) - Circuit with applied QPE operation on
				registers (List) - List containing the u-register as first element, t-register as second element,
									and classical register as final element
		"""
		self.circuit,self.registers = self.initial_state(self.circuit,self.registers)
		self.registers.insert(0,self.t_register)
		U = ControlledTimeEvolutionOperator(self.hamiltonian_list,dt,t)
		return(QPE(self.circuit,self.registers,U))

	def measure_eigenvalues(self,dt,t,E_max):
		"""
		Performs the QPE algorithm and measures the circuit to yield eigenvalue spectra
		Input:
			dt (float) - The time step of the time evolution operation
			t (float) - The evolution time for the time evolution oepration
			E_max (float) - The constant subtracted from Hamiltonian to yield complete eigenvalue spectra
		Output:
			x (numpy array) - array containing the energies measured
			y (numpy array) - array containing the number of times each energy was measured.
		"""
		self.circuit,self.registers = self.__call__(dt,t)
		self.circuit.measure(self.registers[0],self.registers[-1])
		if self.transpile:
			self.circuit = qk.compiler.transpile(self.circuit,backend=self.backend,backend_properties=self.backend.properties(),seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,basis_gates=self.basis_gates,coupling_map=self.coupling_map)
		job = qk.execute(self.circuit, backend = self.backend, shots=self.shots,seed_simulator=self.seed_simulator,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map).result()

		if not self.error_mitigator is None:
			try:
				n_qubits = self.circuit.num_qubits
			except:
				n_qubits = self.circuit.n_qubits
			qubit_list = list(range(len(self.registers[0])))
			meas_filter = self.error_mitigator(n_qubits,qubit_list,self.backend,seed_simulator=self.seed_simulator,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,shots=self.shots)
			result = meas_filter.apply(job)
			result = result.get_counts(0)
		else:
			result = job.get_counts(self.circuit)
		measurements = []
		for key,value in result.items():
			key_ = key[::-1]
			decimal = 0
			for i,bit in enumerate(key_):
				decimal += int(bit)*2**(-i-1)
			if value != 0:
				measurements.append(np.array([E_max-decimal*2*np.pi/t, value]))

		measurements = np.array(measurements)
		x = measurements[:,0]
		indexes = np.argsort(x)
		x = x[indexes]
		y = measurements[:,1]
		y = y[indexes]
		return(x,y)

	def find_peaks(self,x,y,min_measure=15):
		"""
		Finds the estimated eigenvalue and variance by averaging the peaks
		input:
			x (array) - x output from measure_eigenvalues
			y (array) - y output from measure_eigenvalues
			min_measure (int) - Minimum measurements of state before it is considered
			for eigenvalue estimation.
		output:
			eigenvalues (list) - Estimated eigenvalues
			varEigs (list) - Estimated variance of eigenvalue approximation
		"""
		eigenvalues = []
		var_eigenvalues = []
		xi_list = []
		minMeasBool = False
		sumxiyi = 0
		sumyi = 0
		for xi, yi in zip(x,y):
			if yi >= min_measure:
				minMeasBool = True
				sumxiyi += xi*yi
				sumyi += yi
				xi_list.append(xi)
			if minMeasBool and yi < min_measure:
				minMeasBool = False
				mu = sumxiyi/sumyi
				eigenvalues.append(mu)
				sumxiyi=0
				sumyi = 0
				var = 0
				for val in xi_list:
					var += (val - mu)**2
				var/= len(xi_list)
				var_eigenvalues.append(var)
				xi_list = []
		eigenvalues = np.array(eigenvalues)
		var_eigenvalues = np.array(var_eigenvalues)
		return(eigenvalues,var_eigenvalues)