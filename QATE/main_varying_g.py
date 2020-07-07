from QATE import *
import matplotlib.pylab as plt
from utils import *

n_fermi = 2
n_spin_orbitals=4
factor = 0.2
H_0 = [[factor,[0,'z']],[factor,[1,'z']],[-factor,[2,'z']],[-factor,[3,'z']]]

def initial_state(circuit,registers):
	for i in range(int(len(registers[0])/2)):
		circuit.x(registers[0][i])
	return(circuit,registers)







n = 10
g_array = np.linspace(1,5,n)
E_qate = np.zeros(n)
E_fci = np.zeros(n)
for i,g in enumerate(g_array):
	H_1 = pairing_hamiltonian(n_spin_orbitals,1,g)
	H,E_ref = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),1,g)
	eigvals, eigvecs = np.linalg.eigh(H)
	E_fci[i] = eigvals[0]
	steps=80
	dt = 0.5
	t=steps*dt
	E = []
	for k in range(1,80):
		solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
		E.append(solver.calculate_energy(early_stopping=k))
	solver = QATE(n_spin_orbitals,H_0,H_1,initial_state,dt,t,seed_simulator=42)
	E.append(solver.calculate_energy(80))
	E = np.array(E)
	E_qate[i] = np.min(E)
np.save('QATE_2F_4O_g_80.npy',E_qate)

E_qate = np.load('QATE_2F_4O_g_80.npy')

fig1 = plt.figure(1)

frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(g_array,E_qate,'r+',label='QATE')
plt.plot(g_array,E_fci,label='FCI')
plt.title(r'Ideal QATE. {} particles, {} spin orbitals, $\delta=${},dt={},steps=55'.format(n_fermi,n_spin_orbitals,1,55))
plt.ylabel('Energy [u.l]')
plt.legend()


#Residual plot
difference = np.abs(E_qate- E_fci)
frame2=fig1.add_axes((.1,.1,.8,.2))       
plt.plot(g_array,difference,'.')
plt.ylabel(r'|$E_{FCI} - E_{QATE}$|')
plt.xlabel('g')
plt.grid()
plt.show()