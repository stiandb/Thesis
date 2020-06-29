import sys
sys.path.append('../')
from utils import *
from VQE import *
from hamiltonian import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map, ibmq_london
import matplotlib.pylab as plt

np.random.seed(46)
seed_simulator=46

n_fermi = 2
n_spin_orbitals = 4
delta=1
g=1
t = np.random.randn(int(n_fermi/2)*int(( n_spin_orbitals  - n_fermi)/2))

H,e = hamiltonian(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
eigvals,eigvecs = np.linalg.eigh(H)
print(eigvals)

uccd_ansatz = PairingSimpleUCCDAnsatz(PairingInitialState(n_fermi),dt=1,T=1)
hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,delta,g)
solver = VQE(hamiltonian_list,uccd_ansatz,n_spin_orbitals,shots=1000,seed_simulator=42,noise_model=noise_model,basis_gates=basis_gates,transpile=True,optimization_level=3,seed_transpiler=42,coupling_map=coupling_map,error_mitigator=ErrorMitigation())
solver.classical_optimization(t,method='Powell',max_fev=40)
E_noisy = solver.energies
x_noisy = list(range(1,len(E_noisy)+1))

uccd_ansatz = pairing_ansatz
hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,delta,g)
solver = VQE(hamiltonian_list,uccd_ansatz,n_spin_orbitals,shots=1000,seed_simulator=seed_simulator,noise_model=noise_model,basis_gates=basis_gates,transpile=True,optimization_level=3,seed_transpiler=42,coupling_map=coupling_map,error_mitigator=ErrorMitigation())
solver.classical_optimization(t,method='Powell',max_fev=40)
E_noisy1 = solver.energies
x_noisy1 = list(range(1,len(E_noisy1)+1))

uccd_ansatz = PairingSimpleUCCDAnsatz(PairingInitialState(n_fermi),dt=1,T=1)
solver = VQE(hamiltonian_list,uccd_ansatz,n_spin_orbitals,shots=1000,seed_simulator=seed_simulator)
solver.classical_optimization(t,method='Powell',max_fev=40)
E_ideal = solver.energies
x_ideal = list(range(1,len(E_ideal)+1))




n = len(E_noisy) if (len(E_noisy) > len(E_ideal)) else len(E_ideal)
E_ci = np.ones(n)*eigvals[0]
x_ci = list(range(1,n+1))

plt.plot(x_ideal,E_ideal,label='Ideal UCCD')
plt.plot(x_noisy,E_noisy,label='Noisy UCCD')
plt.plot(x_noisy1,E_noisy1,label='Noisy Rotation Ansatz')
plt.plot(x_ci,E_ci,'_',label='CI energy')
plt.xlabel('Function evaluations')
plt.ylabel('Energy [u.l]')
plt.legend()
plt.title(r'VQE on Pairing model with UCCD ansatz. {} particles, {} spin orbitals, $\delta = ${}, $g = ${}.'.format(n_fermi,n_spin_orbitals,delta,g))
plt.show()
