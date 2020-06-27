import sys
sys.path.append('../')
from utils import *

class HamiltonianSimulation:
	def __init__(self,u_qubits,t_qubits,hamiltonian_list,initial_state):
		self.hamiltonian_list = hamiltonian_list
		self.u_qubits = u_qubits
		self.t_qubits = t_qubits
		self.t_register = qk.QuantumRegister(t_qubits)
		self.circuit,self.registers= initialize_circuit(u_qubits,1,t_qubits)
		self.circuit.add_register(self.t_register)
		self.initial_state = initial_state

	def __call__(self,dt,t):
		self.circuit,self.registers = self.initial_state(self.circuit,self.registers)
		self.registers.insert(0,self.t_register)
		U = ControlledTimeEvolutionOperator(self.hamiltonian_list,dt,t)
		return(QPE(self.circuit,self.registers,U))

	def measure_eigenvalues(self,dt,t,E_max,shots=1000,backend=qk.Aer.get_backend('qasm_simulator'),seed_simulator=None,noise_model=None,basis_gates=None):
		self.circuit,self.registers = self.__call__(dt,t)
		self.circuit.measure(self.registers[0],self.registers[-1])
		job = qk.execute(self.circuit, backend = backend, shots=shots,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates)
		result = job.result()
		result = result.get_counts(self.circuit)

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