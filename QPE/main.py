import sys
sys.path.append('../')
from utils import *
from matplotlib.pylab import *
from HamiltonianSimulation import *
from settings import ibmq_16_melbourne_noise_model as noise, ibmq_16_melbourne_basis_gates as gate, ibmq_16_melbourne_coupling_map as Map

np.random.seed(10202)


u_qubits = 4

delta = 1
g = 1
hamiltonian_list = pairing_hamiltonian(u_qubits,delta,g)
E_max = 2
hamiltonian_list[-1][0] -= E_max

H1, e_ref= PairingFCIMatrix()(1,2,1,1)
eigvals1,eigvecs = np.linalg.eigh(H1)

H1,e_ref = PairingFCIMatrix()(2,2,1,1)
eigvals,eigvecs = np.linalg.eigh(H1)
xticksx = [eigvals1[0],0,eigvals[0],eigvals1[1]]


dt = 0.005
t = 100*dt
for t_qubits in [4,6,8]:
	for noise_model,basis_gates,coupling_map in [[noise,gate,Map],[None,None,None]]:
		if not noise_model is None:
			if t_qubits  > 4:
				continue
			error_mitigator = ErrorMitigation()
		else:
			error_mitigator = None
		solver = HamiltonianSimulation(u_qubits,t_qubits,hamiltonian_list,qpe_pairing_initial_state,seed_simulator=42,noise_model=noise_model,error_mitigator=error_mitigator)
		x,y = solver.measure_eigenvalues(dt,t,E_max)
		if noise_model is None:
			string = 'Ideal'
		else:
			string = 'Noisy'
		np.save('x{}{}.npy'.format(t_qubits,string),x)
		np.save('y{}{}.npy'.format(t_qubits,string),y)
		

t_qubits = 4
string = 'Ideal'
x4 = np.load('x{}{}.npy'.format(t_qubits,string))
y4 = np.load('y{}{}.npy'.format(t_qubits,string))
t_qubits = 6
string = 'Ideal'
x6 = np.load('x{}{}.npy'.format(t_qubits,string))
y6 = np.load('y{}{}.npy'.format(t_qubits,string))
t_qubits = 8
string = 'Ideal'
x8 = np.load('x{}{}.npy'.format(t_qubits,string))
y8 = np.load('y{}{}.npy'.format(t_qubits,string))
solver = HamiltonianSimulation(u_qubits,t_qubits,hamiltonian_list,qpe_pairing_initial_state)
print(solver.find_peaks(x8,y8,2))
print(xticksx)
plt.plot(x4,y4,'b--',alpha=0.5,label='4 t-qubits')
plt.plot(x6,y6,'g--',alpha=0.5,label='6 t-qubits')
plt.plot(x8,y8,'r',label='8 t-qubits')
plt.title(string + r' QPE: {} u-qubits, $\delta = ${}, $g = ${}'.format(u_qubits,delta,g))
plt.xlabel('Energy [u.l]')
plt.xticks(xticksx)
plt.xlim(min(xticksx) - 1, max(xticksx) + 1)
plt.ylabel('Times measured')
plt.legend()
plt.show()




