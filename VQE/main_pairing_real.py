import sys
sys.path.append('../')
from utils import *
from VQE import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
import matplotlib.pylab as plt
from qiskit import IBMQ
IBMQ.load_account()
my_provider = IBMQ.get_provider()
backend  = my_provider.get_backend('ibmq_london')
np.random.seed(42)



n_fermi = 2
n_spin_orbitals = 4

n = 4
res = np.zeros((n,3,2,2))
g_array = np.linspace(0,10,n)
delta = 1
for j, error_mitigator in enumerate([ErrorMitigation()]):
	for k,ansatz in enumerate([SimplePairingAnsatz()]):
		for i,g in enumerate(g_array):
			H,e = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
			eigvals,eigvecs = np.linalg.eigh(H)
			res[i,1,0,0] = eigvals[0]
			theta = np.random.randn(1)
			hamiltonian_list= pairing_hamiltonian(n_spin_orbitals,delta,g)
			solver = VQE(hamiltonian_list,ansatz,n_spin_orbitals,shots=1000,seed_simulator=42,seed_transpiler=42,transpile=True,optimization_level=3,basis_gates=basis_gates,coupling_map=coupling_map,error_mitigator=None,print_energies=True)
			solver.classical_optimization(theta)
			solver.backend=backend
			solver.error_mitigator=error_mitigator
			E = solver.expectation_value(solver.theta)
			res[i,2,k,j] = E
			np.save('temprr{}{}{}.npy'.format(j,k,i),res)
			print(100*(i+1)/n,'%',j,k)
res[:,0,0,0] = g_array




fig1 = plt.figure(1)

frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(res[:,0,0,0],res[:,1,0,0],label='FCI')
plt.plot(res[:,0,0,0],res[:,2,0,0],'r+',label='VQE: Error mitigation')
plt.title(r'IBMQ London VQE: Simple ansatz. {} particles, {} spin orbitals, $\delta = {}$.'.format(n_fermi,n_spin_orbitals,delta))
plt.ylabel('Energy [u.l]')
plt.legend()


#Residual plot
difference1 = np.abs(res[:,1,0,0] - res[:,2,0,0])


frame2=fig1.add_axes((.1,.1,.8,.2))       
plt.plot(res[:,0,0,0],difference1,'r+')
plt.ylabel(r'|$E_{FCI} - E_{VQE}$|')
plt.xlabel('g')
plt.grid()
plt.show()
