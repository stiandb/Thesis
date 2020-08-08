from AutoEncoder import *
from QATE import *
import matplotlib.pylab as plt
from utils import *
from qiskit.circuit.random import random_circuit

n_qubits=  3
n_parts = 20
depth_per_sub_circuit = 10
circuit_inverse = []
seed = 47

full_circuit = random_circuit(n_qubits,depth_per_sub_circuit,seed=seed)
circuit_inverse.append(full_circuit.inverse())
for i in range(n_parts-1):
	seed += 1
	circuit_temp = random_circuit(n_qubits,depth_per_sub_circuit,seed=seed)
	full_circuit += circuit_temp 
	circuit_inverse.append(circuit_temp.inverse())

loss = []
print('Depth of full circuit: ', full_circuit.depth())
print('Depth of sub circuits: ', depth_per_sub_circuit)
for shots in [1000,10000]:
	ansatz = EulerRotationAnsatz(linear_entangler)
	quantum_register = qk.QuantumRegister(n_qubits,'q')
	classical_register = qk.ClassicalRegister(1)
	ancilla_register = qk.QuantumRegister(1)
	circuit_temp = qk.QuantumCircuit(quantum_register,ancilla_register,classical_register)
	inner = np.zeros(n_parts)
	for k in range(n_parts):
		def circuit_inv(theta,circuit_,registers_):
			if k != 0:
				ansatz_temp = EulerRotationAnsatz(linear_entangler,inverse=True)
				circuit_ += circuit_inverse[k].copy()
				circuit_,registers_ = ansatz_temp(w,circuit_,registers_)
			else:
				circuit_ += circuit_inverse[k].copy()
			return(circuit_,registers_)
		encoder = AutoEncoder(ansatz,circuit_inv,n_qubits=n_qubits,n_weights=3*3*n_qubits,shots=shots,seed_simulator=42)
		w = encoder.fit(0,print_loss=True,method='Powell')
		loss.append(min(encoder.loss_train))
		circuit_temp += circuit_inverse[k].copy().inverse()
		print('Depth reconstructed step ',k,': ', circuit_temp.depth())
		circuit = qk.QuantumCircuit(quantum_register,ancilla_register,classical_register)
		registers = [quantum_register,ancilla_register,classical_register]
		circuit,registers = ansatz(w,circuit,registers)
		print('Ansatz depth: ', circuit.depth())
		circuit += circuit_temp.copy().inverse()
		for i in range(len(registers[0])):
			circuit.x(registers[0][i])
		circuit.mcrx(np.pi,[registers[0][i] for i in range(len(registers[0]))],ancilla_register[0])
		circuit.measure(ancilla_register,registers[-1])
		job = qk.execute(circuit, backend = qk.Aer.get_backend('qasm_simulator'),seed_simulator=42,shots=shots)
		job = job.result()
		result = job.get_counts(circuit)
		inner_product = 0
		for key,value in result.items():
			if key == '1':
				inner_product += value
		inner_product /= shots
		inner[k] = inner_product
		print('Step ',k,' Inner product: ',inner_product)
	np.save('loss_min{}.npy'.format(shots),np.array(loss))
	np.save('inner_product{}.npy'.format(shots),inner)


inner = np.load('inner_product1000.npy')
inner2 = np.load('inner_product10000.npy')
plt.plot(range(depth_per_sub_circuit,depth_per_sub_circuit*n_parts+1,depth_per_sub_circuit),np.sqrt(inner),'r+',label='1000 measurements')
plt.plot(range(depth_per_sub_circuit,depth_per_sub_circuit*n_parts+1,depth_per_sub_circuit),np.sqrt(inner2),'g+',label='10000 measurements')
axes1 = plt.gca()
axes2 = axes1.twiny()

axes2.set_xticks(range(0,n_parts+2))

axes1.set_xlabel('Depth of random circuit')
axes2.set_xlabel("Step of recursive algorithm")
axes1.set_ylabel(r'|Inner Product|')
axes1.legend()
plt.title('Recursive circuit optimization of random circuit')
plt.show()



