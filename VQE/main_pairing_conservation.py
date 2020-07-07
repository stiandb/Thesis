import sys
sys.path.append('../')
from utils import *
from VQE import *
from hamiltonian import *
import matplotlib.pylab as plt

n_fermi = 4
n_spin_orbitals = 6



delta = 1
g = 1
pair_number_op = pair_number_operator(n_spin_orbitals)
particle_number_op = particle_number_operator(n_spin_orbitals)
lamb = 1e-1
regularization = [[int(n_fermi/2),lamb,pair_number_op],[n_fermi,lamb,particle_number_op]]
H,e = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
eigvals,eigvecs = np.linalg.eigh(H)
E_FCI_sol = eigvals[0]
print('Solution: ',E_FCI_sol)

H,e = PairingFCIMatrix()(1,int(n_spin_orbitals/2),delta,g)
eigvals,eigvecs = np.linalg.eigh(H)
E_FCI_min = eigvals[0]
print('Minima: ',E_FCI_min)

theta = np.random.randn(3*n_spin_orbitals*3)
hamiltonian_list= pairing_hamiltonian(n_spin_orbitals,delta,g)
solver = VQE(hamiltonian_list,EulerRotationAnsatz(linear_entangler),n_spin_orbitals,shots=500,seed_simulator=42,print_energies=True,regularization=regularization)
solver.classical_optimization(theta)
E = solver.energies
E_reg = solver.energies_regularized

#res = np.load('pairing_uccd_4_8_1_g.npy')
np.save('pairing_regularization_4_6_1_1_lamb01.npy',E_reg)
np.save('pairing_2_6_1_1_lamb01.npy',E)

if E.shape[0] > E_reg.shape[0]:
	n = E.shape[0]
else:
	n = E_reg.shape[0]

E_fci_solution = np.ones(n)*E_FCI_sol
E_fci_minima = np.ones(n)*E_FCI_min

plt.plot(E,'g',label='VQE Loss')
plt.plot(E_reg,'r',label='VQE Loss + Regularization')
plt.plot(E_fci_solution,label='FCI Solution')
plt.plot(E_fci_minima,label='FCI Minima')
plt.title(r'Regularized VQE. {} particles, {} spin orbitals, $\delta = {}$, g = {}, $\lambda = ${}'.format(n_fermi,n_spin_orbitals,delta,g,lamb))
plt.xlabel('Function evaluations')
plt.ylabel('Loss')
plt.legend()
plt.show()