import sys
sys.path.append('../')
from utils import *
from matplotlib.pylab import *
from HamiltonianSimulation import *
from settings import ibmq_16_melbourne_noise_model as noise, ibmq_16_melbourne_basis_gates as gate, ibmq_16_melbourne_coupling_map as Map

np.random.seed(10202)


u_qubits = 4
t_qubits = 6#8

delta = 1
g = 1
hamiltonian_list = pairing_hamiltonian(u_qubits,delta,g)
E_max = 2
hamiltonian_list[-1][0] -= E_max

dt = 0.005
t = 50*dt#100*dt

for noise_model,basis_gates,coupling_map in [[noise,gate,Map],[None,None,None]]:
	if not noise_model is None:
		error_mitigator = ErrorMitigation()
		transpile=True
	else:
		error_mitigator = None
		transpile=False
	solver = HamiltonianSimulation(u_qubits,t_qubits,hamiltonian_list,pairing_initial_state,seed_simulator=42,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,transpile=transpile,seed_transpiler=42,optimization_level=3,error_mitigator=error_mitigator)
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


