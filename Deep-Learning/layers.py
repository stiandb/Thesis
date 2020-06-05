import numpy as np
import sys
sys.path.append('../')
from utils import *


class Linear:
	def __init__(self,n_inputs=None,n_outputs=None,bias=True,shots=500,eps = 1e-8):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.w = np.random.randn(n_outputs,(n_inputs + 1) if bias else n_inputs)
		self.shots = shots
		self.bias = bias
		self.w_size = self.n_outputs*((n_inputs + 1) if bias else n_inputs)
		self.eps = eps

	def set_weights(self,w,w_idx):
		n_inputs = (self.n_inputs + 1) if self.bias else self.n_inputs
		self.w = w[w_idx:(w_idx+self.n_outputs*n_inputs)].reshape(self.n_outputs,n_inputs)
		w_idx += self.n_outputs*n_inputs
		return(w_idx)

	def __call__(self,x):
		out = np.zeros(self.n_outputs)
		self.inputs = x
		if self.bias:
			x_ = np.zeros(x.shape[0] + 1)
			x_[0] = 1/np.sqrt(2)
			x_[1:] = x/((np.sqrt(np.sum(x**2))*np.sqrt(2))+self.eps)
		else:
			x_ = x[i,:]
		for node in range(self.w.shape[0]):
			n_qubits = int(np.ceil(np.log2(x_.shape[0])))
			amplitude_register = qk.QuantumRegister(n_qubits)
			classical_register = qk.ClassicalRegister(1)
			circuit = qk.QuantumCircuit(amplitude_register,classical_register)
			registers = [amplitude_register,classical_register]
			measurement = squared_inner_product(x_,self.w[node,:],circuit,registers)
			out[node] = measurement
		self.activation = out
		return(out)



class RNN:
	def __init__(self,n_inputs=None,n_hidden=None,bias=True,shots=500,eps=1e-8):
		self.wx = np.random.randn(n_hidden,(n_inputs + 1) if bias else n_inputs)+0.1
		self.wh = np.random.randn(n_hidden,(n_hidden + 1) if bias else n_hidden)+0.1
		self.shots = shots
		self.bias = bias
		self.eps = eps
		self.n_hidden = n_hidden
		self.n_inputs = n_inputs
		self.w_size = n_hidden*((n_inputs + 1) if bias else n_inputs) + n_hidden*((n_hidden + 1) if bias else n_hidden)
		self.n_outputs = self.n_hidden

	def set_weights(self,w,w_idx):
		n_inputs = (self.n_inputs + 1) if self.bias else self.n_inputs
		self.wx = w[w_idx:(w_idx+self.n_hidden*n_inputs)].reshape(self.n_hidden,n_inputs)
		w_idx += self.n_hidden*n_inputs
		n_hidden = (self.n_hidden + 1) if self.bias else self.n_hidden
		self.wh = w[w_idx:(w_idx+self.n_hidden*n_hidden)].reshape(self.n_hidden,n_hidden)
		w_idx += self.n_hidden*n_hidden
		return(w_idx)

	def __call__(self,x,h_0):
		timesteps = x.shape[0]
		h = np.zeros((timesteps,h_0.shape[0]))
		for i in range(x.shape[0]):
			if self.bias:
				x_i = np.zeros(x.shape[1] + 1)
				h_i = np.zeros(h_0.shape[0] + 1)
				x_i[0] = 1/np.sqrt(2)
				x_i[1:] = x[i,:]/((np.sqrt(np.sum(x[i,:]**2))*np.sqrt(2))+self.eps)
				h_i[0] = 1/np.sqrt(2)
				h_i[1:] = h_0/((np.sqrt(np.sum(h_0**2))*np.sqrt(2)) + self.eps)
			else:
				x_i = x[i,:]
				h_i = h_0
			xh = np.concatenate((x_i,h_i))
			h_temp = np.zeros(h_0.shape)
			for node in range(h_0.shape[0]):
				wx = self.wx[i,:]
				wh = self.wh[i,:]
				w = np.concatenate((wx,wh))
				n_qubits = int(np.ceil(np.log2(xh.shape[0])))
				amplitude_register = qk.QuantumRegister(n_qubits)
				classical_register = qk.ClassicalRegister(1)
				circuit = qk.QuantumCircuit(amplitude_register,classical_register)
				registers = [amplitude_register,classical_register]
				measurement = squared_inner_product(xh,w,circuit,registers,shots=self.shots)
				h_temp[node] = measurement
			h[i,:] = h_temp
			h_0 = h_temp
		return(h)

class AnsatzLayer:
	def __init__(self,n_inputs=None,n_outputs=None,n_weights=None,ansatz=None,shots=500):
		self.shots = shots
		self.n_inputs = n_inputs
		self.n_qubits = int(np.ceil(np.log2(n_inputs)))
		self.n_outputs = n_outputs
		self.n_weights = n_weights
		self.w = np.random.randn(n_outputs,n_weights)
		self.ansatz=ansatz
		self.w_size = n_outputs*n_weights
		self.shots=shots

	def set_weights(self,w,w_idx):
		self.w = w[w_idx:(w_idx+self.n_outputs*self.n_weights)].reshape(self.n_outputs,self.n_weights)
		w_idx += self.n_outputs*self.n_weights
		return(w_idx)

	def __call__(self,x):
		output = np.zeros(self.n_outputs)
		for i in range(self.n_outputs):
			amplitude_register = qk.QuantumRegister(self.n_qubits)
			classical_register = qk.ClassicalRegister(1)
			ancilla_register = qk.QuantumRegister(1)
			circuit = qk.QuantumCircuit(amplitude_register,ancilla_register,classical_register)
			registers = [amplitude_register,ancilla_register,classical_register]
			encoder = AmplitudeEncoder()
			circuit, registers = encoder(circuit,registers,x)
			circuit,registers = self.ansatz(self.w[i,:],circuit,registers)
			circuit.mcrx(np.pi,[registers[0][i] for i in range(len(registers[0]))],registers[1][0])
			circuit.measure(registers[1],registers[-1])
			job = qk.execute(circuit, backend = qk.Aer.get_backend('qasm_simulator'), shots=self.shots)
			result = job.result().get_counts(circuit)
			out = 0
			for key,value in result.items():
				if key == '1':
					out += value
			out /= self.shots
			output[i] = out
		return(output)
			

"""
class LinearParallel:
	def __init__(self,n_inputs=None,n_outputs=None,bias=True,shots=10000):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.w = np.random.randn(n_outputs,(n_inputs + 1) if bias else n_inputs)
		self.shots = shots
		self.bias = bias
		self.w_size = self.n_outputs*((n_inputs + 1) if bias else n_inputs)

	def set_weights(self,w,w_idx):
		n_inputs = (self.n_inputs + 1) if self.bias else self.n_inputs
		self.w = w[w_idx:(w_idx+self.n_outputs*n_inputs)].reshape(self.n_outputs,n_inputs)
		w_idx += self.n_outputs*n_inputs
		return(w_idx)

	def __call__(self,x):
		out = np.zeros(self.n_outputs)
		self.inputs = x
		n_qubits = int(np.ceil(np.log2(self.w.shape[0]*self.w.shape[1])))

		n_input_qubits = int(np.ceil(np.log2(x.shape[0]+(1 if self.bias else 0))))
		if 2**(n_qubits - n_input_qubits) < self.n_outputs:
			layer = Linear(n_inputs=self.n_inputs,n_outputs=self.n_outputs)
			layer.w = self.w 
			out = layer(x)
		else:
			encoder = QuantumAmplitudeEncoding(self.w.flatten(),n_classical=self.n_outputs)
			qc,qb,qa,cb = encoder.encode()
			ancilla1 = qk.QuantumRegister(self.n_outputs)
			ancilla2 = qk.QuantumRegister(self.n_outputs)
			qc.add_register(ancilla1)
			qc.add_register(ancilla2)
			enc = QuantumAmplitudeEncoding()
			enc.n_qubits = n_qubits - n_input_qubits

			binary_list = enc.convert_binary(range(2**(n_qubits-n_input_qubits)))
			binary_list = binary_list[:self.n_outputs]
			
			for idx,binary_str in enumerate(binary_list):
				for j,bit in enumerate(binary_str):
					if bit == '0':
						qc.x(qb[j])
				qc.mcrx(np.pi,[qb[i] for i in range(n_qubits-n_input_qubits)],ancilla1[idx])
				for j, bit in enumerate(binary_str):
					if bit == '0':
						qc.x(qb[j])

			encoder.make_amplitude_list(self.inputs,intercept=self.bias)
			for i in range(n_qubits-n_input_qubits):
				qc.h(qb[i])

			qc,qb,qa,cb = encoder.encode(qc,qb,qa,cb,inverse=False)
			qc.x([qb[i] for i in range(n_qubits)])
			qc.x([ancilla1[i] for i in range(self.n_outputs)])
			for j in range(self.n_outputs):
				qc.x(ancilla1[j])
				qubit_list = [qb[i] for i in range(n_qubits)]
				qubit_list.extend([ancilla1[i] for i in range(self.n_outputs)])
				qc.mcrx(np.pi,qubit_list,ancilla2[j])
				qc.x(ancilla1[j])
			qc.measure(ancilla2,cb)
			job = qk.execute(qc, backend = qk.Aer.get_backend('qasm_simulator'), shots=self.shots)
			result = job.result().get_counts(qc)
			
			for key,value in result.items():
				key1 = key[::-1]
				for node,bit in enumerate(key1):
					if bit == '1':
						out[node] += value
			out = out/self.shots
		self.activation = out
		return(out)
"""

