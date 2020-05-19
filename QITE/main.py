import sys
sys.path.append('../')
from QITE import *
from utils import *



alpha = np.sqrt(1/2)
beta = alpha
hamiltonian_list = [[alpha,[0,'x']],[beta,[0,'z']]]
a_list = [[1,[0,'x']],[1,[0,'y']],[1,[0,'z']],[1]]
a_list = [a_list for i in range(len(hamiltonian_list))]

X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
H = alpha*X +beta*Z
eigvals,eigvecs = np.linalg.eigh(H)
print(eigvals)
print(eigvecs)

def H_initial_state(circuit,registers):
	#circuit.ry(-np.pi/4,registers[0][0])
	return(circuit,registers)

dt = 0.1
solver = QITE(1,hamiltonian_list,a_list,dt,H_initial_state,shots=1000,lamb=0.1)
solver.solve(100)

"""
n_fermi = 2
n_spin_orbitals = 4
hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,1,1)
t = np.ones(int(n_fermi /2)*int(( n_spin_orbitals  - n_fermi)/2))
a_temp = PairingUCCD(n_fermi,n_spin_orbitals,t).hamiltonian_list
a_list = [a_temp for i in range(len(hamiltonian_list))]
dt = 0.005
solver = QITE(n_spin_orbitals,hamiltonian_list,a_list,dt,pairing_initial_state,shots=1000)
solver.solve(100)
print(solver.measure_energy())
"""