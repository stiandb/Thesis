from QATE import *

n_spin_orbitals=4
factor = 0.2
H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]
H_1 = pairing_hamiltonian(n_spin_orbitals,1,1)

def initial_state(circuit,registers):
	for i in range(int(len(registers[0])/2)):
		circuit.x(registers[0][i])
	return(circuit,registers)



"""dt = 0.1
t=100
steps = int(t/dt)
print(steps)

solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
for k in range(100,steps,100):
	print(solver.calculate_energy(early_stopping=k))"""

steps=100
dt = 0.2
t=steps*dt




for k in range(10,steps,10):
	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
	print(solver.calculate_energy(early_stopping=k))


solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
print(solver.calculate_energy())
