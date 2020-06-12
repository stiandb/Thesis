import sys
sys.path.append('../')
from utils import *
from VQE import *
from hamiltonian import *
np.random.seed(42)
n_fermi = 4
n_spin_orbitals = 8
t = np.random.randn(int(n_fermi /2)*int(( n_spin_orbitals  - n_fermi)/2))
H,e = hamiltonian(2,4,1,5)
eigvals,eigvecs = np.linalg.eigh(H)
print(eigvals)
uccd_ansatz = PairingUCCD(n_fermi,n_spin_orbitals,t,dt=0.33333333,T=1)
hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,1,5)
solver = VQE(hamiltonian_list,uccd_ansatz,n_spin_orbitals,ancilla=1,shots=1000)
solver.classical_optimization(t,method='Nelder-Mead')

