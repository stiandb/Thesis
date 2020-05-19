import sys
sys.path.append('../')
from utils import *
from matplotlib.pylab import *

u_qubits = 4
t_qubits = 8
u_register = qk.QuantumRegister(u_qubits)
t_register = qk.QuantumRegister(t_qubits)
ancilla_register = qk.QuantumRegister(1)
classical_register = qk.ClassicalRegister(t_qubits)
circuit = qk.QuantumCircuit(t_register,u_register,ancilla_register,classical_register)

for i in range(0,u_qubits,2):
	circuit.h(u_register[i])
	circuit.cx(u_register[i],u_register[i+1])


registers = [t_register,u_register,ancilla_register,classical_register]


delta = 1
g = 1
hamiltonian_list = pairing_hamiltonian(u_qubits,delta,g)
E_max = 2
hamiltonian_list[-1][0] -= E_max

dt = 0.005
t = 100*dt
U = ControlledTimeEvolutionOperator(hamiltonian_list,dt,t)

circuit, registers = QPE(circuit,registers,U)


circuit.measure(t_register,classical_register)
job = qk.execute(circuit, backend = qk.Aer.get_backend('qasm_simulator'), shots=1000)
result = job.result()
result = result.get_counts(circuit)

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
plt.plot(x,y)
plt.show()
