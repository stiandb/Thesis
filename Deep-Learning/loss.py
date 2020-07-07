import numpy as np 
import sys
sys.path.append('../')
from utils import *
from sklearn.metrics import log_loss
from copy import deepcopy

class MSE:
	def __call__(self,y_pred,y):
		return(np.mean((y_pred.flatten() - y.flatten())**2))

class binary_cross_entropy:
	def __call__(self,y_pred,y):
		y_p = y_pred.copy()
		y_p[y_p == 1] = 1 - 1e-14
		y_p[y_p == 0] = 1e-14
		return( np.mean(-np.log(y_p)*y - np.log(1 - y_p)*(1 - y) ))

class cross_entropy:
	def __call__(self,y_pred,y):
		return(log_loss(y,y_pred))

class rayleigh_quotient:
	def __init__(self,H):
		self.H = H

	def __call__(self,x,*args):
		H = self.H
		x = x.T
		x -= 0.5
		return((x.T@H@x/(x.T@x)).flatten()[0])

class eigenvector_ode:
	def __init__(self,A,dt):
		self.dt = dt
		self.A = A

	def __call__(self,x,*args):
		loss = 0
		x = x[0,:,:]
		for t in range(x.shape[0]-1):
			loss += np.mean(( (x[t+1,:] - x[t,:])/self.dt + self.A@x[t+1,:] - (x[t+1,:].T@self.A@x[t+1,:])*x[t+1,:] )**2)
		quotient = rayleigh_quotient(self.A)
		print('Quotient: ',quotient(x[-1,:]))
		return(loss)

class SubsetAutoEncoderInnerProduct:
	def __init__(self,U_subenc,shots=1000,seed_simulator=42):
		self.U_subenc = U_subenc
		self.shots = shots
		self.seed_simulator=seed_simulator

	def __call__(self,theta,x,circuit,registers):
		loss = 0
		loss_list = []
		for sample in range(x.shape[0]):
			circuit,register = self.U_subenc(theta[sample,:],circuit,registers)
			encoder = AmplitudeEncoder(inverse=True)
			circuit,registers = encoder(x[sample,:],circuit,registers)
			ancilla_register = qk.QuantumRegister(1)
			circuit.add_register(ancilla_register)
			registers.insert(1,ancilla_register)
			for i in range(len(registers[0])):
				circuit.x(registers[0][i])
			circuit.mcrx(np.pi,[registers[0][i] for i in range(len(registers[0]))],ancilla_register[0])
			circuit.measure(ancilla_register,registers[-1])
			job = qk.execute(circuit, backend = qk.Aer.get_backend('qasm_simulator'), shots=self.shots,seed_simulator=self.seed_simulator).result()
			result = job.get_counts(circuit)
			inner_product = 0
			for key,value in result.items():
				if key == '1':
					inner_product += value
			inner_product /= self.shots
			loss += inner_product
			loss_list.append(inner_product)
		return(-loss/x.shape[0],loss_list)


class UnitaryComparison:
	def __init__(self,U_1,U_2,n_qubits=None,shots=1000,seed_simulator=42):
		self.U_1 = U_1
		self.U_2 = U_2
		self.n_qubits=n_qubits
		self.shots=shots
		self.seed_simulator=seed_simulator
	def __call__(self,theta_1,theta_2,initial_state):
		n_qubits = self.n_qubits
		q_reg = qk.QuantumRegister(n_qubits)
		c_reg = qk.ClassicalRegister(1)
		circuit = qk.QuantumCircuit(q_reg,c_reg)
		registers = [q_reg,c_reg]
		circuit,registers = initial_state(circuit,registers,inverse=False)
		circuit,registers = deepcopy(self.U_1)(theta_1,circuit,registers)
		circuit,registers = deepcopy(self.U_2)(theta_2,circuit,registers)
		circuit,registers = initial_state(circuit,registers,inverse=True)
		ancilla_register = qk.QuantumRegister(1)
		circuit.add_register(ancilla_register)
		registers.insert(1,ancilla_register)
		for i in range(len(registers[0])):
			circuit.x(registers[0][i])
		circuit.mcrx(np.pi,[registers[0][i] for i in range(len(registers[0]))],ancilla_register[0])
		circuit.measure(ancilla_register,registers[-1])
		job = qk.execute(circuit, backend = qk.Aer.get_backend('qasm_simulator'), shots=self.shots,seed_simulator=self.seed_simulator).result()
		result = job.get_counts(circuit)
		inner_product = 0
		for key,value in result.items():
			if key == '1':
				inner_product += value
		inner_product /= self.shots
		del circuit,registers,ancilla_register
		return(-inner_product)








