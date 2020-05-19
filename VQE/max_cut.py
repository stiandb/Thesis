import sys
sys.path.append('../')
from utils import *
from VQE import *


W = np.array([[0,3,0,0,1],[3,0,2,0,3],[0,2,0,2,0],[0,0,2,0,3],[1,3,0,3,0]])

hamiltonian_list = max_cut_hamiltonian(W)

n_qubits = 5
solver = VQE(hamiltonian_list,y_rotation_ansatz,n_qubits)

theta = np.random.randn(n_qubits)
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