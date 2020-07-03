from QATE import *
import matplotlib.pylab as plt
from utils import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map


"""n_spin_orbitals=4
factor = 0.2
H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]
H_1 = pairing_hamiltonian(n_spin_orbitals,1,1)

def initial_state(circuit,registers):
	for i in range(int(len(registers[0])/2)):
		circuit.x(registers[0][i])
	return(circuit,registers)




steps=80
dt = 0.4
t=steps*dt



E = []



for k in range(1,steps,3):
	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
	E.append(solver.calculate_energy(early_stopping=k))


solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
E.append(solver.calculate_energy())
E = np.array(E)"""

E = np.load('QATE_2F_4O_range1_steps_3.npy')

plt.plot(E)
plt.show()

"""def initial_state(circuit,registers):
	for i in range(int(len(registers[0])/2)):
		circuit.x(registers[0][i])
	return(circuit,registers)


n_spin_orbitals=4
factor = 0.2
n = 10
steps=40
dt = 0.4
t=steps*dt
E_ideal = np.zeros(n)
E_noisy = np.zeros(n)
E_fci = np.zeros(n)
for i,g in enumerate(np.linspace(0.5,5,n)):
	H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]
	H_1 = pairing_hamiltonian(n_spin_orbitals,1,g)
	H,E_ref = PairingFCIMatrix()(1,2,1,g)
	eigvals,eigvecs = np.linalg.eigh(H)
	E_fci[i] = eigvals[0]
	E_temp_ideal = []
	E_temp_noisy = []
	for k in range(1,steps,3):
		solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
		E_temp_ideal.append(solver.calculate_energy(early_stopping=k))
		solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42,seed_transpiler=42,transpile=True,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,error_mitigator=ErrorMitigation())
		E_temp_noisy.append(solver.calculate_energy(early_stopping=k))


	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
	E_temp_ideal.append(solver.calculate_energy())
	E_ideal[i] = min(E_temp_ideal)
	E_noisy[i] = min(E_temp_noisy)


np.save('qate_ideal.npy',E_ideal)
np.save('qate_noisy.npy',E_noisy)
np.save('qate_fci.npy',E_fci)
"""
