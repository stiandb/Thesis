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

	def measure_eigenvalues(self,dt,t,E_max,shots=1000,backend=qk.Aer.get_backend('qasm_simulator')):
		self.circuit,self.registers = self.__call__(dt,t)
		self.circuit.measure(self.registers[0],self.registers[-1])
		job = qk.execute(self.circuit, backend = backend, shots=shots)
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