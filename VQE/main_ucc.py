import sys
sys.path.append('../')
from utils import *
from VQE import *

n_fermi = 2
n_spin_orbitals = 4
t = np.random.randn(int(n_fermi /2)*int(( n_spin_orbitals  - n_fermi)/2))

uccd_ansatz = PairingUCCD(n_fermi,n_spin_orbitals,t)
hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,1,1)
solver = VQE(hamiltonian_list,uccd_ansatz,n_spin_orbitals,ancilla=1)
solver.classical_optimization(t,method='Powell')

