import sys
sys.path.append('../')
sys.path.append('../Deep-Learning')
from VQE import *
from utils import *
from layers import *
import numpy as np
np.random.seed(42)
seed_simulator = 42


y_rotation = YRotation(bias=True)

def ansatz(theta,circuit,registers,classical_bits):
	ans = AnsatzRotationLinear(n_inputs=4,n_outputs=5,n_weights_r=3,n_weights_a=2,ansatz=y_rotation_ansatz,rotation=y_rotation,classical_bits=classical_bits,n_parallel=5)
	ans.set_weights(theta,0)
	circuit,registers = ans(np.ones((1,4)))
	return(circuit,[registers[1],registers[-1]])


W = np.array([[0,3,0,0,1],[3,0,2,0,3],[0,2,0,2,0],[0,0,2,0,3],[1,3,0,3,0]])

hamiltonian_list = max_cut_hamiltonian(W)

n_qubits = 5
solver = VQE(hamiltonian_list,ansatz,n_qubits,seed_simulator=seed_simulator)

theta = np.random.randn(5*3 + 2)
theta = solver.classical_optimization(theta,method='Powell')

ansatz_register = qk.QuantumRegister(n_qubits)
classical_register = qk.ClassicalRegister(n_qubits)
circuit= qk.QuantumCircuit(ansatz_register,classical_register)
registers = [ansatz_register,classical_register]

circuit,registers = y_rotation_ansatz(theta,circuit,registers)

circuit.measure(registers[0],registers[1])
job = qk.execute(circuit, backend = qk.Aer.get_backend('qasm_simulator'), shots=1000)
result = job.result()
result = result.get_counts(circuit)
print(result)
