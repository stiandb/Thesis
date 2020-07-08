import sys
sys.path.append('../')
sys.path.append('../CCD-Pairing')
from utils import *
from VQE import *
from hamiltonian import *
import matplotlib.pylab as plt
from CCD_pairing import *

n_fermi = 4
n_spin_orbitals = 8

n = 10
res = np.zeros((n,4))
g_array = np.linspace(1,5,n)
delta = 1


for i,g in enumerate(g_array):
	H,e = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
	eigvals,eigvecs = np.linalg.eigh(H)
	res[i,1] = eigvals[0]
	E_ccd,E_ccd_ref = run(4,4,delta,g,alpha=0.4)
	res[i,3] = E_ccd + E_ccd_ref
	theta = np.random.randn(int(n_fermi/2)*int((n_spin_orbitals-n_fermi)/2))
	hamiltonian_list= pairing_hamiltonian(n_spin_orbitals,delta,g)
	solver = VQE(hamiltonian_list,PairingUCCD(n_fermi,n_spin_orbitals,PairingInitialState(n_fermi),theta,dt=1,T=1),n_spin_orbitals,shots=10000,seed_simulator=42,print_energies=False)
	solver.classical_optimization(theta)
	E = solver.expectation_value(solver.theta)
	res[i,2] = E
	print(100*(i+1)/n,'%')
res[:,0] = g_array

fig1 = plt.figure(1)

frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(res[:,0],res[:,1],label='FCI')
plt.plot(res[:,0],res[:,2],'r+',label='VQE UCCD')
plt.plot(res[:,0],res[:,3],'g+',label='CCD')
plt.title(r'Ideal VQE. {} particles, {} spin orbitals, $\delta = {}$.'.format(n_fermi,n_spin_orbitals,delta))
plt.ylabel('Energy [u.l]')
plt.legend()


#Residual plot
difference1 = np.abs(res[:,1]- res[:,2])
difference2 = np.abs(res[:,1] - res[:,3])
frame2=fig1.add_axes((.1,.1,.8,.2))       
plt.plot(res[:,0],difference1,'r+')
plt.plot(res[:,0],difference2,'g+')
plt.ylabel(r'|$E_{FCI} - E$|')
plt.xlabel('g')
plt.grid()
plt.show()


