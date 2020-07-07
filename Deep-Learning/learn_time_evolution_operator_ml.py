import sys
sys.path.append('../')

from utils import *
from AutoEncoder import *
n_fermi = 2
n_spin_orbitals = 4



hamiltonian = pairing_hamiltonian(n_spin_orbitals,1,1)
def TimeEvolution(theta,circuit,registers):
	op = TimeEvolutionOperator(hamiltonian,dt=1,T=1,inverse=True)
	circuit, registers = op.step(circuit,registers)
	circuit.x(registers[0][-1])
	circuit.x(registers[0][-2])
	return(circuit,registers)


q_r = qk.QuantumRegister(4)
c_r = qk.ClassicalRegister(4)
circuit = qk.QuantumCircuit(q_r,c_r)
registers= [q_r,c_r]
circuit,registers = TimeEvolution(0,circuit,registers)
c_depth_TE = circuit.depth()


ansatz = EulerRotationAnsatz(linear_entangler)
encoder = AutoEncoder(ansatz,TimeEvolution,n_qubits=4,n_weights=3*4*3)
w_1 = encoder.fit(0,print_loss=True,method='Powell')
w_1 = encoder.w_opt
q_r = qk.QuantumRegister(4)
c_r = qk.ClassicalRegister(4)
circuit = qk.QuantumCircuit(q_r,c_r)
registers= [q_r,c_r]
circuit,registers = EulerRotationAnsatz(linear_entangler)(w_1,circuit,registers)
print('Ansatz circuit depth: ',circuit.depth())
print('Time Evolution circuit depth:',c_depth_TE)



loss_1 = encoder.loss_train
print('Min loss: ',np.sqrt(-np.min(loss_1)))

plt.plot(range(1,loss_1.shape[0]+1,50),np.sqrt(-loss_1[0:-1:50]))
plt.title('Autoencoding of time evolution circuit')
plt.xlabel('Function evaluations')
plt.ylabel('|Inner Product|')
plt.show()
