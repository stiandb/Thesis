import sys
sys.path.append('../')
from utils import *
from VQE import *
from hamiltonian import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
import matplotlib.pylab as plt

np.random.seed(46)
seed_simulator=46

n_fermi = 2
n_spin_orbitals = 4
n = 30


E_fci = np.zeros(n)
E_simple_noisy = np.zeros(n)
E_uccd_noisy = np.zeros(n)
E_uccd_ideal = np.zeros(n)


for i,angle in enumerate(np.linspace(0,2*np.pi,n)):
	theta = np.array([angle])
	g=1
	delta = 1

	t = np.random.randn(int(n_fermi/2)*int(( n_spin_orbitals  - n_fermi)/2))

	H,e = hamiltonian(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
	eigvals,eigvecs = np.linalg.eigh(H)
	E_fci[i] = eigvals[0]
	

	uccd_ansatz = PairingSimpleUCCDAnsatz(PairingInitialState(n_fermi),dt=1,T=1)
	hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,delta,g)
	solver = VQE(hamiltonian_list,uccd_ansatz,n_spin_orbitals,shots=10000,seed_simulator=42+i,noise_model=noise_model,basis_gates=basis_gates,transpile=True,optimization_level=3,seed_transpiler=None,coupling_map=coupling_map,error_mitigator=ErrorMitigation())
	E_uccd_noisy[i] = solver.expectation_value(theta)
	t = np.random.randn(int(n_fermi/2)*int(( n_spin_orbitals  - n_fermi)/2))

	ansatz = simple_pairing_ansatz
	hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,delta,g)
	solver = VQE(hamiltonian_list,ansatz,n_spin_orbitals,shots=10000,seed_simulator=45+i,noise_model=noise_model,basis_gates=basis_gates,transpile=True,optimization_level=3,seed_transpiler=None,coupling_map=coupling_map,error_mitigator=ErrorMitigation())
	E_simple_noisy[i] = solver.expectation_value(theta)

	t = np.random.randn(int(n_fermi/2)*int(( n_spin_orbitals  - n_fermi)/2))
	hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,delta,g)
	uccd_ansatz = PairingSimpleUCCDAnsatz(PairingInitialState(n_fermi),dt=1,T=1)
	solver = VQE(hamiltonian_list,uccd_ansatz,n_spin_orbitals,shots=10000,seed_simulator=49+i)
	E_uccd_ideal[i] = solver.expectation_value(theta)
	print(i+1)

np.save('E_fci_{}_{}.npy'.format(n_fermi,n_spin_orbitals),E_fci)
np.save('E_simple_noisy_{}_{}.npy'.format(n_fermi,n_spin_orbitals),E_simple_noisy)
np.save('E_uccd_noisy_{}_{}.npy'.format(n_fermi,n_spin_orbitals),E_uccd_noisy)
np.save('E_uccd_ideal_{}_{}.npy'.format(n_fermi,n_spin_orbitals),E_uccd_ideal)

plt.plot(E_fci)
plt.plot(E_simple_noisy)
plt.plot(E_uccd_noisy)
plt.plot(E_uccd_ideal)
plt.show()