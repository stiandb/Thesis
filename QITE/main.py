import sys
sys.path.append('../')
from QITE import *
from utils import *
from itertools import product

"""
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

def initial_state(circuit,registers):
	return(circuit,registers)

dt = 0.7
solver = QITE(1,hamiltonian_list,a_list,dt,initial_state,shots=1000,lamb=20)
solver.solve(100)
"""


g0 = 1
g1 = 1
g2 = 1
g3 = 1
g4 = 1
g5 = 1
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-np.complex(0,1)],[np.complex(0,1),0]])
I = np.array([[1,0],[0,1]])
H = g0 + g1*np.kron(Z,I) + g2*np.kron(I,Z) + g3*np.kron(Z,Z) + g4*np.kron(X,X) + g5*np.kron(Y,Y)
vals, vecs = np.linalg.eigh(H)
print(vals)
n_spin_orbitals = 2
hamiltonian_list = [[g0],[g1,[0,'z']],[g2,[1,'z']],[g3,[0,'z'],[1,'z']],[g4,[0,'x'],[1,'x']],[g5,[0,'y'],[1,'y']]]
a_list_temp = list(product('xyzI',repeat=n_spin_orbitals))
a_list = []
for tup in a_list_temp:
	gate_1 = tup[0]
	gate_2 = tup[1]
	if gate_1 == 'I' and gate_2 != 'I':
		a_list.append([1,[1,gate_2]])
	if gate_2 == 'I' and gate_1 != 'I':
		a_list.append([1,[0,gate_1]])
	if gate_1 == 'I' and gate_2 == 'I':
		a_list.append([1])
	if gate_1 != 'I' and gate_2 != 'I':
		a_list.append([1,[0,gate_1],[1,gate_2]])
a_list = [a_list for i in range(len(hamiltonian_list))]
dt = 0.1
lamb = 20
def initial_state(circuit,registers):
	n = len(registers[0])
	for i in range(n):
		circuit.ry(1.5,registers[0])
	return(circuit,registers)
solver = QITE(n_spin_orbitals,hamiltonian_list,a_list,dt,initial_state,lamb=lamb)
solver.solve(100)



n_fermi = 2
n_spin_orbitals = 4
hamiltonian_list = pairing_hamiltonian(n_spin_orbitals,1,1)
t = np.random.randn(int(n_fermi /2)*int(( n_spin_orbitals  - n_fermi)/2))
pairing_uccd = PairingUCCD(n_fermi,n_spin_orbitals,t).hamiltonian_list
a_list = [ [[1,[2,'z']],[1,[3,'z']],[1,[2,'z'],[3,'z']]], [[1,[2,'z']],[1,[3,'z']],[1,[2,'z'],[3,'z']]], \
			[[1,[0,'z']],[1,[1,'z']],[1,[0,'z'],[1,'z']]], [[1,[0,'z']],[1,[1,'z']],[1,[0,'z'],[1,'z']]], \
			[[1,[0,'z']],[1,[1,'z']],[1,[0,'z'],[1,'z']]], \
			[[1, [0, 'x'], [1, 'x'], [2, 'x'], [3, 'x']], [1, [0, 'x'], [1, 'x'], [2, 'y'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'x'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'x'], [2, 'x'], [3, 'y']], [1, [0, 'y'], [1, 'x'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'x'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'y'], [3, 'y']] ],\
			[[1, [0, 'x'], [1, 'x'], [2, 'x'], [3, 'x']], [1, [0, 'x'], [1, 'x'], [2, 'y'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'x'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'x'], [2, 'x'], [3, 'y']], [1, [0, 'y'], [1, 'x'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'x'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'y'], [3, 'y']] ],\
			[[1, [0, 'x'], [1, 'x'], [2, 'x'], [3, 'x']], [1, [0, 'x'], [1, 'x'], [2, 'y'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'x'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'x'], [2, 'x'], [3, 'y']], [1, [0, 'y'], [1, 'x'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'x'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'y'], [3, 'y']] ],\
			[[1, [0, 'x'], [1, 'x'], [2, 'x'], [3, 'x']], [1, [0, 'x'], [1, 'x'], [2, 'y'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'x'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'x'], [2, 'x'], [3, 'y']], [1, [0, 'y'], [1, 'x'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'x'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'y'], [3, 'y']] ],\
			[[1, [0, 'x'], [1, 'x'], [2, 'x'], [3, 'x']], [1, [0, 'x'], [1, 'x'], [2, 'y'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'x'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'x'], [2, 'x'], [3, 'y']], [1, [0, 'y'], [1, 'x'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'x'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'y'], [3, 'y']] ],\
			[[1, [0, 'x'], [1, 'x'], [2, 'x'], [3, 'x']], [1, [0, 'x'], [1, 'x'], [2, 'y'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'x'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'x'], [2, 'x'], [3, 'y']], [1, [0, 'y'], [1, 'x'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'x'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'y'], [3, 'y']] ],\
			[[1, [0, 'x'], [1, 'x'], [2, 'x'], [3, 'x']], [1, [0, 'x'], [1, 'x'], [2, 'y'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'x'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'x'], [2, 'x'], [3, 'y']], [1, [0, 'y'], [1, 'x'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'x'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'y'], [3, 'y']] ],\
			[[1, [0, 'x'], [1, 'x'], [2, 'x'], [3, 'x']], [1, [0, 'x'], [1, 'x'], [2, 'y'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'x'], [3, 'y']], [1, [0, 'x'], [1, 'y'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'x'], [2, 'x'], [3, 'y']], [1, [0, 'y'], [1, 'x'], [2, 'y'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'x'], [3, 'x']], [1, [0, 'y'], [1, 'y'], [2, 'y'], [3, 'y']] ],\
			[[1,[2,'z']],[1,[3,'z']],[1,[2,'z'],[3,'z']]], [[1,[2,'z']],[1,[3,'z']],[1,[2,'z'],[3,'z']]], \
			[[1,[2,'z']],[1,[3,'z']],[1,[2,'z'],[3,'z']]], [[1],[1,[0,'z']]] ]
a_list = [pairing_uccd for i in range(len(hamiltonian_list))]

dt = 0.1
solver = QITE(n_spin_orbitals,hamiltonian_list,a_list,dt,pairing_initial_state,shots=1000,lamb=0.5)
solver.solve(100)
print(solver.measure_energy())

