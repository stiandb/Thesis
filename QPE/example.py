import sys
sys.path.append('../')
from utils import *
from matplotlib.pylab import *
from HamiltonianSimulation import *



np.random.seed(10202)

u_qubits = 2
t_qubits = 8

hamiltonian_list = [[0.25,[0,'x']],[0.25,[1,'x']],[-0.5]]
def initial_state(circuit,registers):
	return(circuit,registers)
E_max = 2
hamiltonian_list[-1][0] -= E_max

dt = 0.005
t = 100*dt

solver = HamiltonianSimulation(u_qubits,t_qubits,hamiltonian_list,initial_state)
x,y = solver.measure_eigenvalues(dt,t,E_max,seed_simulator=42)

plt.plot(x,y)
plt.xlabel('Energy')
plt.ylabel('Times Measured')
plt.title('QPE Example on two-qubit Hamiltonian')
plt.show()
