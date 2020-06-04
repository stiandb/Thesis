from QATE import *

n_fermi=2
n_spin_orbitals=4
factor = 0.13
H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]
H_1 = pairing_hamiltonian(n_spin_orbitals,1,1)

def initial_state(circuit,registers):
	for i in range(int(len(registers[0])/2)):
		circuit.x(registers[0][i])
	return(circuit,registers)

dt = 0.1
for t in range(50,60,1):
	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t)
	print(solver.calculate_energy())