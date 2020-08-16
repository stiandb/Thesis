from QATE import *
import matplotlib.pylab as plt
from utils import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map

n_fermi = 2
n_spin_orbitals=4
factor = 0.2
H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]
H_1 = pairing_hamiltonian(n_spin_orbitals,1,5)

def initial_state(circuit,registers):
	for i in range(int(len(registers[0])/2)):
		circuit.x(registers[0][i])
	return(circuit,registers)


steps=80
dt = 0.5
t=steps*dt
E = []
for k in range(1,steps):
	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
	E.append(solver.calculate_energy(early_stopping=k))


solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
E.append(solver.calculate_energy())
E = np.array(E)

np.save('QATE_2F_4O_range1_steps05.npy',E)
E = np.load('QATE_2F_4O_range1_steps05.npy')
H,E_ref = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),1,1)
eigvals, eigvecs = np.linalg.eigh(H)
E_fci = np.ones(steps)*eigvals[0]
plt.plot(list(range(1,steps+1)),E,'r+',label='QATE')
plt.plot(list(range(1,steps+1)),E_fci,label='FCI')
plt.title(r'QATE: {} particles, {} spin orbitals, $\delta = 1$, $g = 1$, dt = {}, {} steps'.format(n_fermi,n_spin_orbitals,dt,steps))
plt.xlabel('Trotter step')
plt.ylabel('Energy [u.l]')
plt.legend()
plt.show()

H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]
H_1 = pairing_hamiltonian(n_spin_orbitals,1,5)


steps=55
dt = 0.2
t=steps*dt

E = []
for k in range(1,steps):
	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42,shots=10000)
	E.append(solver.calculate_energy(early_stopping=k))


solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
E.append(solver.calculate_energy(steps))
E = np.array(E)

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

H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]
H_1 = pairing_hamiltonian(n_spin_orbitals,1,5)

steps=55
dt = 0.2
stop = 21
t=steps*dt

E = []
for k in range(1,stop):
	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation(),noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map)
	E.append(solver.calculate_energy(early_stopping=k))


solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation(),noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map)
E.append(solver.calculate_energy(stop))
E = np.array(E)
np.save('QATENoisemodel.npy',E)

E = np.load('QATENoisemodel.npy')

H,E_ref = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),1,5)
eigvals, eigvecs = np.linalg.eigh(H)
E_fci = np.ones(E.shape[0])*eigvals[0]
plt.plot(list(range(1,E.shape[0]+1)),E,'r+',label='QATE')
plt.plot(list(range(1,E_fci.shape[0]+1)),E_fci,label='FCI')
plt.title(r'Noisy QATE: {} particles, {} spin orbitals, $\delta = 1$, $g = 5$, dt = {}, {} steps'.format(n_fermi,n_spin_orbitals,dt,steps))
plt.xlabel('Trotter step')
plt.ylabel('Energy [u.l]')
plt.legend()
plt.show()



