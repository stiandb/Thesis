import sys
sys.path.append('../Deep-Learning')
from AutoEncoder import *
from QATE import *
import matplotlib.pylab as plt
from utils import *

shots=1000

n_fermi = 2
n_spin_orbitals=4
factor = 0.2
H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]
H_1 = pairing_hamiltonian(n_spin_orbitals,1,5)


delta = 1
g = 5


steps=55
dt = 0.2
t=steps*dt
E = []

for k in range(0,steps):
	ansatz= YRotationAnsatz(linear_entangler,inverse=False)
	def TE(theta,circuit,registers):
		solver = QATE(n_spin_orbitals,H_0,H_1,identity_circuit,dt,t,seed_simulator=42,shots=shots)
		circuit,registers = solver.trotter_step(circuit,registers,k,inverse=True)
		if k == 0:
			for i in range(int(len(registers[0])/2)):
				circuit.x(registers[0][i])
		else:
			ansatz_temp =  YRotationAnsatz(linear_entangler,inverse=True)
			circuit,registers = ansatz_temp(w,circuit,registers)
		return(circuit,registers)
	encoder = AutoEncoder(ansatz,TE,n_qubits=4,n_weights=4,shots=shots)
	w = encoder.fit(0,print_loss=True,method='Powell')
	def init_state(circuit,registers):
		return(ansatz(w,circuit,registers))
	solver = QATE(n_spin_orbitals,H_0,H_1,init_state,dt,t,seed_simulator=42,shots=shots)
	E_i = solver.calculate_energy(0)
	E.append(E_i)
	print(k, E_i)
E = np.array(E)
np.save('QATE_recursive_learning{}.npy'.format(shots),E)

E = np.load('QATE_recursive_learning{}.npy'.format(shots))
H,E_ref = PairingFCIMatrix()(1,2,1,5)
eigvals,eigvecs = np.linalg.eigh(H)
E_FCI = eigvals[0]
plt.plot(range(1,steps+1),E,'r+',label='Learned Time Evolution Operator')
plt.plot(range(1,steps+1),np.ones(steps)*E_FCI,label='FCI')
plt.title(r'Recursive Circuit Optimization. {} particles, {} spin orbitals, $\delta = ${}, $g = ${}'.format(n_fermi,n_spin_orbitals,delta,g))
plt.xlabel('Step')
plt.ylabel('Energy [u.l]')
plt.legend()
plt.show()


	

