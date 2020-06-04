from utils import *

X = np.random.randn(8)
X = X/np.sqrt(np.sum(X**2))
print(X)
n_qubits = int(np.ceil(np.log2(X.shape[0])))
amplitude_register = qk.QuantumRegister(n_qubits)
classical_register = qk.ClassicalRegister(n_qubits)
circuit = qk.QuantumCircuit(amplitude_register,classical_register)
registers = [amplitude_register,classical_register]

encoder = AmplitudeEncoder(eps=0)
circuit,registers = encoder(circuit,registers,X)
shots = 1000000
circuit.measure(registers[0],registers[-1])
job = qk.execute(circuit, backend = qk.Aer.get_backend('qasm_simulator'), shots=shots)
result = job.result().get_counts(circuit)
for key,value in result.items():
	print(key,np.sqrt(value/shots))


n = 3
x = np.random.randn(3)
w = np.random.randn(3)
shots=10000
x_ = x/np.sqrt(np.sum(x**2))
w_ = w/np.sqrt(np.sum(w**2))
n_qubits = int(np.ceil(np.log2(X.shape[0])))
amplitude_register = qk.QuantumRegister(n_qubits)
classical_register = qk.ClassicalRegister(1)
circuit = qk.QuantumCircuit(amplitude_register,classical_register)
registers = [amplitude_register,classical_register]

print(np.sum(w_*x_)**2)
print(squared_inner_product(x,w,circuit,registers,shots))