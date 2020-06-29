import sys
sys.path.append('../')
from utils import *
from matplotlib.pylab import *
from HamiltonianSimulation import *
from settings import melbourne_noise_model, melbourne_basis_gates

np.random.seed(10202)


u_qubits = 4
t_qubits = 8

delta = 1
g = 1
hamiltonian_list = pairing_hamiltonian(u_qubits,delta,g)
E_max = 2
hamiltonian_list[-1][0] -= E_max

dt = 0.005
t = 100*dt

for noise_model,basis_gates in zip([melbourne_noise_model,None],[melbourne_basis_gates,None]):
	solver = HamiltonianSimulation(u_qubits,t_qubits,hamiltonian_list,pairing_initial_state,seed_simulator=42,noise_model=noise_model,basis_gates=basis_gates)
	x,y = solver.measure_eigenvalues(dt,t,E_max)
	if noise_model is None:
		string = 'Ideal'
	else:
		string = 'Noisy'
	plt.plot(x,y)
	plt.title(string + r' QPE: {} u-qubits, {} t-qubits, $\delta = ${}, $g = ${}'.format(u_qubits,t_qubits,delta,g))
	plt.xlabel('Energy [u.l]')
	plt.ylabel('Times measured')
	plt.show()


