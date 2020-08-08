from utils import *
from AutoEncoder import *
import numpy as np

class RecursivePairingUCCD:
	def __init__(self,n_fermi,n_spin_orbitals,initial_state,t,dt=1,T=1,steps=None,ansatz=None,n_weights=None,shots=1000,print_loss=False):
		"""
		Input:
			n_fermi (int) - The number of particles in the system
			n_spin_orbitals (int) - The number of spin orbitals in the system
			t (numpy array) - 1D array with the UCCD amplitudes
			dt (float) - Time step in the trotter approximation
			T (float) - Total time in the trotter approximation
		"""
		self.n_fermi = n_fermi
		self.n_spin_orbitals = n_spin_orbitals
		self.hamiltonian_list = []
		self.dt = dt
		self.T = T
		self.initial_state=initial_state
		self.doubles_operator(t)
		self.steps = steps
		self.ansatz = ansatz
		self.n_weights = n_weights
		self.shots=shots
		self.print_loss = print_loss
		self.t_prev = None


	def __call__(self,t,circuit,registers):
		"""
		Input:
			t (numpy array) - 1D array with the UCCD amplitudes
			circuit (qiskit QuantumCircuit) - The circuit to apply the UCCD ansatz to
			registers (list) - List containing qiskit registers. The first register should be the one to apply the ansatz to.
								the second to last should be an ancilla register
								the final should be the classical register
		Outputs:
			circuit (qiskit QuantumCircuit) - The circuit with an applied UCCD ansatz
			registers (list) - List containing the corresponding registers
		"""
		n_fermi = self.n_fermi
		n_spin_orbitals = self.n_spin_orbitals
		self.replace_parameters(t)
		steps = self.steps
		terms_per_step = int(np.ceil(len(self.hamiltonian_list)/steps))
		if not self.t_prev is None:
			comparison = self.t_prev == t
			if comparison.all():
				circuit,registers = self.ansatz(self.w,circuit,registers)
				return(circuit,registers)

		for i in range(steps):
			time_evo_list = self.hamiltonian_list[i*terms_per_step:i*terms_per_step + terms_per_step]
			if len(time_evo_list) == 0:
				continue
			def ansatz_2(theta,circ,reg):
				time_evolution = TimeEvolutionOperator(time_evo_list,self.dt,self.T,inverse=True)
				circ,reg = time_evolution(circ,reg)
				if i == 0:
					circ,reg = self.initial_state(circ,reg)
				else:
					self.ansatz.inverse = True
					circ,reg= self.ansatz(w,circ,reg)
					self.ansatz.inverse= False
				return(circ,reg)
			encoder = AutoEncoder(self.ansatz,ansatz_2,n_qubits=self.n_spin_orbitals,n_weights=self.n_weights,shots=self.shots,seed_simulator=42)
			w = encoder.fit(0,print_loss=self.print_loss,method='Powell')
		self.w = w
		circuit,registers = self.ansatz(w,circuit,registers)
		self.t_prev = t
		return(circuit,registers)

	
	def doubles_operator(self,t):
		"""
		Calculates the doubles operator for arbitrary amplitudes t
		"""
		n_fermi = self.n_fermi
		n_spin_orbitals = self.n_spin_orbitals
		idx=0
		for i in range(0,n_fermi,2):
			for a in range(n_fermi,n_spin_orbitals,2):
				self.hamiltonian_list.append([t[idx]/8,[i,'x'],[i+1,'x'],[a,'y'],[a+1,'x']])
				self.hamiltonian_list.append([t[idx]/8,[i,'y'],[i+1,'x'],[a,'y'],[a+1,'y']])
				self.hamiltonian_list.append([t[idx]/8,[i,'x'],[i+1,'y'],[a,'y'],[a+1,'y']])
				self.hamiltonian_list.append([t[idx]/8,[i,'x'],[i+1,'x'],[a,'x'],[a+1,'y']])
				self.hamiltonian_list.append([-t[idx]/8,[i,'y'],[i+1,'x'],[a,'x'],[a+1,'x']])
				self.hamiltonian_list.append([-t[idx]/8,[i,'x'],[i+1,'y'],[a,'x'],[a+1,'x']])
				self.hamiltonian_list.append([-t[idx]/8,[i,'y'],[i+1,'y'],[a,'y'],[a+1,'x']])
				self.hamiltonian_list.append([-t[idx]/8,[i,'y'],[i+1,'y'],[a,'x'],[a+1,'y']])
				idx+=1


	def replace_parameters(self,t):
		"""
		Replaces the amplitudes in hamiltonian_list with new amplitudes t
		"""
		idx = 0
		for i in range(0,self.n_fermi,2):
			for a in range(self.n_fermi,self.n_spin_orbitals,2):
				
				self.hamiltonian_list[idx][0] = t[int(idx/8)]/8
				self.hamiltonian_list[idx+1][0] = t[int(idx/8)]/8
				self.hamiltonian_list[idx+2][0] = t[int(idx/8)]/8
				self.hamiltonian_list[idx+3][0] = t[int(idx/8)]/8
				self.hamiltonian_list[idx+4][0] = -t[int(idx/8)]/8
				self.hamiltonian_list[idx+5][0] = -t[int(idx/8)]/8
				self.hamiltonian_list[idx+6][0] = -t[int(idx/8)]/8
				self.hamiltonian_list[idx+7][0] = -t[int(idx/8)]/8
				idx += 8


