import sys
sys.path.append('../')
from utils import *
from VQE import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
import matplotlib.pylab as plt

n_fermi = 2
n_spin_orbitals = 4

n = 40
res = np.zeros((n,3,2,2))
g_array = np.linspace(0,10,n)
delta = 1
for j, error_mitigator in enumerate([None,ErrorMitigation()]):
	for k,ansatz in enumerate([SimplePairingAnsatz(),PairingSimpleUCCDAnsatz(PairingInitialState(n_fermi))]):
		for i,g in enumerate(g_array):
			H,e = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
			eigvals,eigvecs = np.linalg.eigh(H)
			res[i,1,0,0] = eigvals[0]
			theta = np.random.randn(1)
			hamiltonian_list= pairing_hamiltonian(4,delta,g)
			solver = VQE(hamiltonian_list,ansatz,n_spin_orbitals,shots=10000,seed_simulator=42,seed_transpiler=42,transpile=True,optimization_level=3,noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,error_mitigator=error_mitigator)
			solver.classical_optimization(theta)
			E = solver.expectation_value(solver.theta)
			res[i,2,k,j] = E
			print(100*(i+1)/n,'%',j,k)
res[:,0,0,0] = g_array


fig1 = plt.figure(1)

frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(res[:,0,0,0],res[:,1,0,0],label='FCI')
plt.plot(res[:,0,0,0],res[:,2,1,0],'r+',label='VQE: No error mitigation')
plt.plot(res[:,0,0,0],res[:,2,1,1],'g+',label='VQE: Error mitigation')
plt.title(r'Noisy VQE: UCCD ansatz. {} particles, {} spin orbitals, $\delta = {}$.'.format(n_fermi,n_spin_orbitals,delta))
plt.ylabel('Energy [u.l]')
plt.legend()


#Residual plot
difference1 = np.abs(res[:,1,0,0]- res[:,2,1,0])
difference2 = np.abs(res[:,1,0,0]- res[:,2,1,1])

frame2=fig1.add_axes((.1,.1,.8,.2))       
plt.plot(res[:,0,0,0],difference1,'r+')
plt.plot(res[:,0,0,0],difference2,'g+')
plt.ylabel(r'|$E_{FCI} - E_{VQE}$|')
plt.xlabel('g')
plt.grid()
plt.show()

fig1 = plt.figure(1)

frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(res[:,0,0,0],res[:,1,0,0],label='FCI')
plt.plot(res[:,0,0,0],res[:,2,0,0],'r+',label='VQE: No error mitigation')
plt.plot(res[:,0,0,0],res[:,2,0,1],'g+',label='VQE: Error mitigation')
plt.title(r'Noisy VQE: Simple ansatz. {} particles, {} spin orbitals, $\delta = {}$.'.format(n_fermi,n_spin_orbitals,delta))
plt.ylabel('Energy [u.l]')
plt.legend()


#Residual plot
difference1 = np.abs(res[:,1,0,0]- res[:,2,0,0])
difference2 = np.abs(res[:,1,0,0]- res[:,2,0,1])

frame2=fig1.add_axes((.1,.1,.8,.2))       
plt.plot(res[:,0,0,0],difference1,'r+')
plt.plot(res[:,0,0,0],difference2,'g+')
plt.ylabel(r'|$E_{FCI} - E_{VQE}$|')
plt.xlabel('g')
plt.grid()
plt.show()

plt.plot(res[:,0,0,0],difference2,'r+',label='Simple ansatz')
plt.plot(res[:,0,0,0],np.abs(res[:,1,0,0]- res[:,2,1,1]),'g+',label='UCCD ansatz')
plt.xlabel('g')
plt.ylabel(r'|$E_{FCI} - E_{VQE}$|')
plt.legend()
plt.title(r'Ansatz comparisson. {} particles, {} spin orbitals, $\delta = {}$.'.format(n_fermi,n_spin_orbitals,delta))
plt.show()


