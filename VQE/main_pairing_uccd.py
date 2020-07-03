import sys
sys.path.append('../')
from utils import *
from VQE import *
from hamiltonian import *
import matplotlib.pylab as plt

n_fermi = 4
n_spin_orbitals = 8

n = 10
res = np.zeros((n,3))
g_array = np.linspace(1,5,n)
delta = 1


for i,g in enumerate(g_array):
	H,e = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
	eigvals,eigvecs = np.linalg.eigh(H)
	res[i,1] = eigvals[0]
	theta = np.random.randn(1)
	hamiltonian_list= pairing_hamiltonian(4,delta,g)
	solver = VQE(hamiltonian_list,PairingUCCD(n_fermi,n_spin_orbitals,delta,g),n_spin_orbitals,shots=10000,seed_simulator=42)
	solver.classical_optimization(theta,method='COBYLA')
	E = solver.expectation_value(solver.theta)
	res[i,2] = E
	print(100*(i+1)/n,'%')
res[:,0] = g_array


fig1 = plt.figure(1)

frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(res[:,0],res[:,1],label='FCI')
plt.plot(res[:,0],res[:,2],'r+',label='VQE')
plt.title(r'Ideal VQE. {} particles, {} spin orbitals, $\delta = {}$.'.format(n_fermi,n_spin_orbitals,delta))

plt.ylabel('Energy [u.l]')
plt.legend()