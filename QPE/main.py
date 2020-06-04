import sys
sys.path.append('../')
from utils import *
from matplotlib.pylab import *
from HamiltonianSimulation import *



np.random.seed(10202)

u_qubits = 4
t_qubits = 6

delta = 1
g = 1
hamiltonian_list = pairing_hamiltonian(u_qubits,delta,g)
E_max = 2
hamiltonian_list[-1][0] -= E_max

dt = 0.005
t = 100*dt

solver = HamiltonianSimulation(u_qubits,t_qubits,hamiltonian_list,pairing_initial_state)
x,y = solver.measure_eigenvalues(dt,t,E_max)

plt.plot(x,y)
plt.show()
