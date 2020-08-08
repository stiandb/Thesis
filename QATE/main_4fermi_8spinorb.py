from QATE import *
import matplotlib.pylab as plt
from utils import *
from qiskit import IBMQ



n_fermi = 4
n_spin_orbitals=8
factor = 0.2
H_0 = [[factor,[0,'z']],[factor,[1,'z']],[factor,[2,'z']],[factor,[3,'z']],[-factor,[4,'z']],[-factor,[5,'z']],[-factor,[6,'z']],[-factor,[7,'z']]]
H_1 = pairing_hamiltonian(n_spin_orbitals,1,5)

def initial_state(circuit,registers):
	for i in range(int(len(registers[0])/2)):
		circuit.x(registers[0][i])
	return(circuit,registers)


steps=100
dt = 0.4
t=steps*dt
stop=14
E = []
for k in range(1,stop):
	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
	E.append(solver.calculate_energy(early_stopping=k))

solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
E.append(solver.calculate_energy(stop))
E = np.array(E)
np.save('qate_4_8_1_5.npy',E)
E = np.load('qate_4_8_1_5.npy')

H,E_ref = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),1,5)
eigvals, eigvecs = np.linalg.eigh(H)
E_fci = np.ones(E.shape[0])*eigvals[0]
plt.plot(list(range(1,E.shape[0]+1)),E,'r+',label='QATE')
plt.plot(list(range(1,E_fci.shape[0]+1)),E_fci,label='FCI')
plt.title(r'QATE: {} particles, {} spin orbitals, $\delta = 1$, $g = 5$, dt = {}, {} steps'.format(n_fermi,n_spin_orbitals,dt,steps))
plt.xlabel('Trotter step')
plt.ylabel('Energy [u.l]')
plt.legend()
plt.show()