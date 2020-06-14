from utils import AmplitudeEncoder
import numpy as np
import qiskit as qk
np.random.seed(12)

x = np.random.randn(4)
amplitude_register = qk.QuantumRegister(2)
classical_register = qk.ClassicalRegister(2)
circuit = qk.QuantumCircuit(amplitude_register,classical_register)
registers = [amplitude_register,classical_register]


encoder = AmplitudeEncoder(eps=1e-14)
circuit,registers = encoder(circuit,registers,x,inverse=False)

shots = 1000000
circuit.measure(registers[0],registers[-1])
job = qk.execute(circuit, backend = qk.Aer.get_backend('qasm_simulator'), shots=shots)
result = job.result().get_counts(circuit)
for key,value in result.items():
	print(key,np.sqrt(value/shots))
	
print(x/np.sqrt(np.sum(x**2)))