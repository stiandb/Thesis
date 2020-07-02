import numpy as np
import sys
sys.path.append('../')
from utils import *
from dl_utils import *


class Linear:
	def __init__(self,n_inputs=None,n_outputs=None,bias=True,shots=500,eps = 1e-8,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,seed_transpiler=None,transpile=False,optimization_level=1,error_mitigator=None):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.w = np.random.randn(n_outputs,(n_inputs + 1) if bias else n_inputs)
		self.shots = shots
		self.bias = bias
		self.w_size = self.n_outputs*((n_inputs + 1) if bias else n_inputs)
		self.eps = eps
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.coupling_map=coupling_map
		self.transpile=transpile
		self.optimization_level=optimization_level
		self.seed_transpiler=seed_transpiler
		self.error_mitigator = error_mitigator

	def set_weights(self,w,w_idx=0):
		n_inputs = (self.n_inputs + 1) if self.bias else self.n_inputs
		self.w = w[w_idx:(w_idx+self.n_outputs*n_inputs)].copy().reshape(self.n_outputs,n_inputs)
		w_idx += self.n_outputs*n_inputs
		return(w_idx)

	def __call__(self,x):
		out = np.zeros((x.shape[0],self.n_outputs))
		for sample in range(x.shape[0]):
			if self.bias:
				x_ = np.zeros(x.shape[1] + 1)
				x_[0] = 1/np.sqrt(2)
				x_[1:] = x[sample,:].copy()/((np.sqrt(np.sum(x[sample,:]**2))*np.sqrt(2))+self.eps)
			else:
				x_ = x[sample,:].copy()
			for node in range(self.w.shape[0]):
				n_qubits = int(np.ceil(np.log2(x_.shape[0])))
				amplitude_register = qk.QuantumRegister(n_qubits)
				classical_register = qk.ClassicalRegister(1)
				circuit = qk.QuantumCircuit(amplitude_register,classical_register)
				registers = [amplitude_register,classical_register]
				measurement = squared_inner_product(x_,self.w[node,:],circuit,registers,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,transpile=self.transpile,seed_transpiler=self.seed_transpiler,coupling_map=self.coupling_map,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
				out[sample,node] = measurement
				if not self.seed_simulator is None:
					self.seed_simulator+=1
		return(out)



class RNN:
	def __init__(self,n_predictors=None,n_hidden=None,bias=True,shots=500,eps=1e-8,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,seed_transpiler=None,transpile=False,optimization_level=1,error_mitigator=None):
		self.wx = np.random.randn(n_hidden,(n_predictors + 1) if bias else n_predictors)+0.1
		self.wh = np.random.randn(n_hidden,(n_hidden + 1) if bias else n_hidden)+0.1
		self.shots = shots
		self.bias = bias
		self.eps = eps
		self.n_hidden = n_hidden
		self.n_predictors = n_predictors
		self.w_size = n_hidden*((n_predictors + 1) if bias else n_predictors) + n_hidden*((n_hidden + 1) if bias else n_hidden)
		self.n_outputs = self.n_hidden
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.coupling_map=coupling_map
		self.seed_transpiler=seed_transpiler
		self.transpile=transpile
		self.optimization_level=optimization_level
		self.error_mitigator=error_mitigator

	def set_weights(self,w,w_idx=0):
		n_predictors = (self.n_predictors + 1) if self.bias else self.n_predictors
		self.wx = w[w_idx:(w_idx+self.n_hidden*n_predictors)].copy().reshape(self.n_hidden,n_predictors)
		w_idx += self.n_hidden*n_predictors
		n_hidden = (self.n_hidden + 1) if self.bias else self.n_hidden
		self.wh = w[w_idx:(w_idx+self.n_hidden*n_hidden)].copy().reshape(self.n_hidden,n_hidden)
		w_idx += self.n_hidden*n_hidden
		return(w_idx)

	def __call__(self,x,h_0):
		timesteps = x.shape[1]
		h = np.zeros((x.shape[0],timesteps,h_0.shape[0]))
		for sample in range(x.shape[0]):
			for i in range(x.shape[1]):
				if self.bias:
					x_i = np.zeros(x.shape[2] + 1)
					h_i = np.zeros(h_0.shape[0] + 1)
					x_i[0] = 1/np.sqrt(2)
					x_i[1:] = x[sample,i,:]/((np.sqrt(np.sum(x[sample,i,:]**2))*np.sqrt(2))+self.eps)
					h_i[0] = 1/np.sqrt(2)
					h_i[1:] = h_0/((np.sqrt(np.sum(h_0**2))*np.sqrt(2)) + self.eps)
				else:
					x_i = x[sample,i,:]
					h_i = h_0
				xh = np.concatenate((x_i,h_i))
				h_temp = np.zeros(h_0.shape)
				for node in range(h_0.shape[0]):
					wx = self.wx[node,:]
					wh = self.wh[node,:]
					w = np.concatenate((wx,wh))
					n_qubits = int(np.ceil(np.log2(xh.shape[0])))
					amplitude_register = qk.QuantumRegister(n_qubits)
					classical_register = qk.ClassicalRegister(1)
					circuit = qk.QuantumCircuit(amplitude_register,classical_register)
					registers = [amplitude_register,classical_register]
					measurement = squared_inner_product(xh,w,circuit,registers,shots=self.shots,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
					h_temp[node] = measurement
					if not self.seed_simulator is None:
						self.seed_simulator+=1
				h[sample,i,:] = h_temp
				h_0 = h_temp
		return(h)

class AnsatzLinear:
	def __init__(self,n_inputs=None,n_outputs=None,n_weights=None,ansatz=None,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_inputs = n_inputs
		self.n_qubits = int(np.ceil(np.log2(n_inputs))) if n_inputs > 1 else 1
		self.n_outputs = n_outputs
		self.n_weights = n_weights
		self.w = np.random.randn(n_outputs,n_weights)
		self.ansatz=ansatz
		self.w_size = n_outputs*n_weights
		self.shots=shots
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.coupling_map=coupling_map
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.error_mitigator=error_mitigator

	def set_weights(self,w,w_idx=0):
		self.w = w[w_idx:(w_idx+self.n_outputs*self.n_weights)].copy().reshape(self.n_outputs,self.n_weights)
		w_idx += self.n_outputs*self.n_weights
		return(w_idx)

	def __call__(self,x):
		output = np.zeros((x.shape[0],self.n_outputs))
		for sample in range(x.shape[0]):
			for i in range(self.n_outputs):
				amplitude_register = qk.QuantumRegister(self.n_qubits)
				classical_register = qk.ClassicalRegister(1)
				ancilla_register = qk.QuantumRegister(1)
				circuit = qk.QuantumCircuit(amplitude_register,classical_register)
				registers = [amplitude_register,ancilla_register,classical_register]
				encoder = AmplitudeEncoder()
				circuit, registers = encoder(x[sample,:],circuit,registers)
				circuit.add_register(ancilla_register)
				circuit,registers = self.ansatz(self.w[i,:],circuit,registers)
				circuit.mcrx(np.pi,[registers[0][j] for j in range(len(registers[0]))],registers[1][0])
				circuit.measure(registers[1],registers[-1])
				if self.transpile:
					circuit = qk.compiler.transpile(circuit,backend=self.backend,backend_properties=self.backend.properties(),seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,basis_gates=self.basis_gates,coupling_map=self.coupling_map)
				job = qk.execute(circuit, backend = self.backend,seed_simulator=self.seed_simulator, shots=self.shots,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map).result()
				if not self.error_mitigator is None:
					n_qubits = circuit.n_qubits
					meas_filter = self.error_mitigator(n_qubits,[-1],self.backend,seed_simulator=self.seed_simulator,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,shots=self.shots)
					result = meas_filter.apply(job)
					result = result.get_counts(circuit)
				else:
					result = job.get_counts(circuit)
				out = 0
				for key,value in result.items():
					if key == '1':
						out += value
				out /= self.shots
				output[sample,i] = out
				if not self.seed_simulator is None:
					self.seed_simulator+=1
		return(output)

class AnsatzRNN:
	def __init__(self,n_hidden=None,n_wx=None,n_wh=None,ansatz=None,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_hidden = n_hidden
		self.wx = np.random.randn(n_hidden,n_wx)
		self.wh = np.random.randn(n_hidden,n_wh)
		self.n_wx = n_wx
		self.n_wh = n_wh
		self.n_weights = n_wx + n_wh
		self.ansatz=ansatz
		self.w_size = n_hidden*n_wx + n_hidden*n_wh
		self.shots=shots
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.coupling_map=coupling_map
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.error_mitigator = error_mitigator

	
	def set_weights(self,w,w_idx=0):
		self.wx = w[w_idx:(w_idx+self.n_hidden*self.n_wx)].copy().reshape(self.n_hidden,self.n_wx)
		w_idx += self.n_hidden*self.n_wx
		self.wh = w[w_idx:(w_idx+self.n_hidden*self.n_wh)].copy().reshape(self.n_hidden,self.n_wh)
		w_idx += self.n_hidden*self.n_wh
		return(w_idx)

	def __call__(self,x,h_0):
		timesteps = x.shape[1]
		h = np.zeros((x.shape[0],timesteps,h_0.shape[0]))
		for sample in range(x.shape[0]):
			for t in range(x.shape[1]):
				x_t = x[sample,t,:]
				x_t = x_t.reshape(1,x_t.shape[0])
				x_layer = AnsatzLinear(x_t.shape[1],self.n_hidden,self.n_wx,ansatz=self.ansatz,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
				x_layer.w = self.wx
				h_layer = AnsatzLinear(h_0.shape[0],self.n_hidden,self.n_wh,ansatz=self.ansatz,seed_simulator =self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
				h_layer.w = self.wh
				out1 = x_layer(x_t)
				h_0 = h_0.reshape(1,h_0.shape[0])
				out2 = h_layer(h_0)
				h_temp = (out1 + out2)/2
				h[sample,t,:] = h_temp.flatten()
				h_0 = h_temp.flatten()
				if not self.seed_simulator is None:
					self.seed_simulator+=2
		return(h)

class RotationLinear:
	def __init__(self,n_inputs=None,n_outputs=None,n_weights=None,rotation=None,n_parallel=1,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,classical_bits=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_inputs = n_inputs
		self.n_qubits = int(np.ceil(np.log2(n_inputs))) if n_inputs > 1 else 1
		self.n_outputs = n_outputs
		self.n_weights = n_weights
		self.w = np.random.randn(n_outputs,n_weights)
		self.rotation=rotation
		self.w_size = n_outputs*n_weights
		self.shots = shots
		self.n_parallel = n_parallel
		self.seed_simulator = seed_simulator
		self.backend = backend
		self.noise_model = noise_model
		self.classical_bits=classical_bits
		self.basis_gates=basis_gates
		self.coupling_map=coupling_map
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.error_mitigator=error_mitigator

	def set_weights(self,w,w_idx=0):
		self.w = w[w_idx:(w_idx+self.n_outputs*self.n_weights)].copy().reshape(self.n_outputs,self.n_weights)
		w_idx += self.n_outputs*self.n_weights
		return(w_idx)

	def __call__(self,x):
		output = np.zeros((x.shape[0],self.n_outputs))
		for sample in range(x.shape[0]):
			for i in range(0,self.n_outputs,self.n_parallel):
				if (self.n_outputs - i) < self.n_parallel:
					n_parallel = self.n_outputs - i
				else:
					n_parallel = self.n_parallel
				amplitude_register = qk.QuantumRegister(self.n_qubits)
				if not self.classical_bits is None:
					classical_register = qk.ClassicalRegister(self.classical_bits)
				else:
					classical_register = qk.ClassicalRegister(n_parallel)
				circuit = qk.QuantumCircuit(amplitude_register,classical_register)
				registers = [amplitude_register,classical_register]
				encoder = AmplitudeEncoder()
				circuit, registers = encoder(x[sample,:],circuit,registers)
				ancilla_register = qk.QuantumRegister(n_parallel)
				circuit.add_register(ancilla_register)
				registers.insert(1,ancilla_register)
				for j in range(n_parallel):
					circuit,registers = self.rotation(self.w[i+j,:],j,circuit,registers)
				if not self.classical_bits is None and (self.n_parallel == self.n_outputs):
					return(circuit,registers)
				circuit.measure(registers[1],registers[-1])
				if self.transpile:
					circuit = qk.compiler.transpile(circuit,backend=self.backend,backend_properties=self.backend.properties(),seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,basis_gates=self.basis_gates,coupling_map=self.coupling_map)
				
				job = qk.execute(circuit, backend = self.backend,seed_simulator=self.seed_simulator, shots=self.shots,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map).result()
				if not self.error_mitigator is None:
					n_qubits = circuit.n_qubits
					meas_filter = self.error_mitigator(n_qubits,list(range(n_qubits-n_parallel,n_qubits)),self.backend,seed_simulator=self.seed_simulator,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,shots=self.shots)
					result = meas_filter.apply(job)
					result = result.get_counts(0)
				else:
					result = job.get_counts(circuit)
				out = np.zeros(n_parallel)
				for key,value in result.items():
					key_ = key[::-1]
					for k,qubit in enumerate(key_):
						if qubit == '1':
							out[k] += value
				out /= self.shots
				output[sample,i:(i+n_parallel)] = out
				if not self.seed_simulator is None:
					self.seed_simulator+=1
		return(output)

class RotationRNN:
	def __init__(self,n_hidden=None,n_wx=None,n_wh=None,rotation=None,n_parallel_x=1,n_parallel_h=1,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_hidden = n_hidden
		self.n_outputs = n_hidden
		self.wx = np.random.randn(n_hidden,n_wx)
		self.wh = np.random.randn(n_hidden,n_wh)
		self.n_wx = n_wx
		self.n_wh = n_wh
		self.n_weights = n_wx + n_wh
		self.rotation=rotation
		self.w_size = n_hidden*n_wx + n_hidden*n_wh
		self.shots=shots
		self.n_parallel_x = n_parallel_x
		self.n_parallel_h = n_parallel_h
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model = noise_model
		self.basis_gates=basis_gates
		self.coupling_map=coupling_map
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.error_mitigator=error_mitigator

	
	def set_weights(self,w,w_idx=0):
		self.wx = w[w_idx:(w_idx+self.n_hidden*self.n_wx)].copy().reshape(self.n_hidden,self.n_wx)
		w_idx += self.n_hidden*self.n_wx
		self.wh = w[w_idx:(w_idx+self.n_hidden*self.n_wh)].copy().reshape(self.n_hidden,self.n_wh)
		w_idx += self.n_hidden*self.n_wh
		return(w_idx)

	def __call__(self,x,h_0 = None):
		if h_0 is None:
			h_0 = np.ones(self.n_hidden)
		timesteps = x.shape[1]
		h = np.zeros((x.shape[0],timesteps,h_0.shape[0]))
		for sample in range(x.shape[0]):
			for t in range(x.shape[1]):
				x_t = x[sample,t,:]
				x_t = x_t.reshape(1,x_t.shape[0])
				x_layer = RotationLinear(x_t.shape[1],self.n_hidden,self.n_wx,rotation=self.rotation,n_parallel=self.n_parallel_x,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
				x_layer.w = self.wx
				h_layer = RotationLinear(h_0.shape[0],self.n_hidden,self.n_wh,rotation=self.rotation,n_parallel=self.n_parallel_h,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
				h_layer.w = self.wh
				out1 = x_layer(x_t)
				out2 = h_layer(h_0.reshape(1,h_0.shape[0]))
				h_temp = (out1 + out2)/2
				h[sample,t,:] = h_temp.flatten()
				h_0 = h_temp.flatten()
				if not self.seed_simulator is None:
					self.seed_simulator+=2
		return(h)

class AnsatzRotationLinear:
	def __init__(self,n_inputs=None,n_outputs=None,n_weights_a=None,n_weights_r=None,ansatz=None,rotation=None,n_parallel = 1,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,classical_bits=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_inputs = n_inputs
		self.n_qubits = int(np.ceil(np.log2(n_inputs))) if n_inputs > 1 else 1
		self.n_outputs = n_outputs
		self.w_r = 2*np.pi*np.random.randn(n_outputs,n_weights_r)
		self.n_parallel = n_parallel
		self.n_weights_a = n_weights_a
		self.n_weights_r = n_weights_r
		if n_parallel > 1:
			self.w_a = 2*np.pi*np.random.randn(1,n_weights_a)
			self.w_size = n_outputs*n_weights_r + n_weights_a
		else:
			self.w_a = 2*np.pi*np.random.randn(n_outputs,n_weights_a)
			self.w_size = n_outputs*n_weights_a + n_weights_r*n_outputs
		self.ansatz=ansatz
		self.rotation=rotation
		self.shots=shots
		self.seed_simulator = seed_simulator
		self.backend = backend
		self.noise_model = noise_model
		self.classical_bits = classical_bits
		self.basis_gates = basis_gates
		self.coupling_map=coupling_map
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.error_mitigator=error_mitigator

	def set_weights(self,w,w_idx=0):
		if self.n_parallel > 1:
			self.w_a = w[w_idx:(w_idx+self.n_weights_a)].copy().reshape(1,self.n_weights_a)
			w_idx += self.n_weights_a
		else:
			self.w_a = w[w_idx:(w_idx+self.n_outputs*self.n_weights_a)].copy().reshape(self.n_outputs,self.n_weights_a)
			w_idx += self.n_outputs*self.n_weights_a
		self.w_r = w[w_idx:(w_idx + self.n_outputs*self.n_weights_r)].copy().reshape(self.n_outputs,self.n_weights_r)
		w_idx += self.n_outputs*self.n_weights_r
		return(w_idx)

	def __call__(self,x):
		output = np.zeros((x.shape[0],self.n_outputs))
		for sample in range(x.shape[0]):
			for i in range(0,self.n_outputs,self.n_parallel):
				if (self.n_outputs - i) < self.n_parallel:
					n_parallel = self.n_outputs - i
				else:
					n_parallel = self.n_parallel
				amplitude_register = qk.QuantumRegister(self.n_qubits)
				if not self.classical_bits is None:
					classical_register = qk.ClassicalRegister(self.classical_bits)
				else:
					classical_register = qk.ClassicalRegister(n_parallel)
				circuit = qk.QuantumCircuit(amplitude_register,classical_register)
				registers = [amplitude_register,classical_register]
				encoder = AmplitudeEncoder()
				circuit, registers = encoder(x[sample,:],circuit,registers)
				ancilla_register = qk.QuantumRegister(n_parallel)
				circuit.add_register(ancilla_register)
				registers.insert(1,ancilla_register)
				if self.n_parallel > 1:
					circuit,registers = self.ansatz(self.w_a[0,:],circuit,registers)
				else:
					circuit,registers = self.ansatz(self.w_a[i,:],circuit,registers)
				for j in range(n_parallel):
					circuit,registers = self.rotation(self.w_r[i+j,:],j,circuit,registers)
				if not self.classical_bits is None and (self.n_parallel == self.n_outputs):
					return(circuit,registers)
				circuit.measure(registers[1],registers[-1])
				if self.transpile:
					circuit = qk.compiler.transpile(circuit,backend=self.backend,backend_properties=self.backend.properties(),seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,basis_gates=self.basis_gates,coupling_map=self.coupling_map)
				job = qk.execute(circuit, backend = self.backend,seed_simulator=self.seed_simulator,shots=self.shots,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map).result()
				if not self.error_mitigator is None:
					n_qubits = circuit.n_qubits
					meas_filter = self.error_mitigator(n_qubits,list(range(n_qubits-n_parallel,n_qubits)),self.backend,seed_simulator=self.seed_simulator,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,shots=self.shots)
					result = meas_filter.apply(job)
					result = result.get_counts(0)
				else:
					result = job.get_counts(circuit)
				out = np.zeros(n_parallel)
				for key,value in result.items():
					key_ = key[::-1]
					for k,qubit in enumerate(key_):
						if qubit == '1':
							out[k] += value
				out /= self.shots
				output[sample,i:(i+n_parallel)] = out
				if not self.seed_simulator is None:
					self.seed_simulator+=1
		return(output)

class AnsatzRotationRNN:
	def __init__(self,n_hidden=None,n_wxa=None,n_wxr=None,n_wha=None,n_whr=None,rotation=None,ansatz=None,n_parallel_x=1,n_parallel_h=1,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_hidden = n_hidden
		self.n_outputs = n_hidden
		self.whr = np.random.randn(n_hidden,n_whr)
		self.wxr = np.random.randn(n_hidden,n_wxr)
		self.n_wxa = n_wxa
		self.n_wha = n_wha
		self.n_wxr = n_wxr
		self.n_whr = n_whr
		self.w_size = self.n_outputs*n_wxr + self.n_outputs*n_whr
		if n_parallel_x > 1:
			self.wxa = np.random.randn(1,n_wxa)
			self.w_size += n_wxa
		else:
			self.wxa = np.random.randn(n_hidden,n_wxa)
			self.w_size += self.n_outputs*n_wxa
		if n_parallel_h > 1:
			self.wha = np.random.randn(1,n_wha)
			self.w_size += n_wha
		else:
			self.wha = np.random.randn(n_hidden,n_wha)
			self.w_size += self.n_outputs*n_wha
		self.rotation=rotation
		self.shots=shots
		self.n_parallel_x = n_parallel_x
		self.n_parallel_h = n_parallel_h
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model = noise_model
		self.basis_gates=basis_gates
		self.ansatz=ansatz
		self.coupling_map=coupling_map
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.error_mitigator=None

	
	def set_weights(self,w,w_idx=0):
		if self.n_parallel_x > 1:
			self.wxa = w[w_idx:(w_idx+self.n_wxa)].copy().reshape(1,self.n_wxa)
			w_idx += self.n_wxa
		else:
			self.wxa = w[w_idx:(w_idx+self.n_outputs*self.n_wxa)].copy().reshape(self.n_outputs,self.n_wxa)
			w_idx += self.n_outputs*self.n_wxa
		self.wxr = w[w_idx:(w_idx + self.n_outputs*self.n_wxr)].copy().reshape(self.n_outputs,self.n_wxr)
		w_idx += self.n_outputs*self.n_wxr
		if self.n_parallel_h > 1:
			self.wha = w[w_idx:(w_idx+self.n_wha)].copy().reshape(1,self.n_wha)
			w_idx += self.n_wha
		else:
			self.wha = w[w_idx:(w_idx+self.n_outputs*self.n_wha)].copy().reshape(self.n_outputs,self.n_wha)
			w_idx += self.n_outputs*self.n_wha
		self.whr = w[w_idx:(w_idx + self.n_outputs*self.n_whr)].copy().reshape(self.n_outputs,self.n_whr)
		w_idx += self.n_outputs*self.n_whr
		return(w_idx)

	def __call__(self,x,h_0 = None):
		if h_0 is None:
			h_0 = np.ones(self.n_hidden)
		timesteps = x.shape[1]
		h = np.zeros((x.shape[0],timesteps,h_0.shape[0]))
		for sample in range(x.shape[0]):
			for t in range(x.shape[1]):
				x_t = x[sample,t,:]
				x_t = x_t.reshape(1,x_t.shape[0])
				x_layer = AnsatzRotationLinear(x_t.shape[1],self.n_hidden,n_weights_a=self.n_wxa,n_weights_r=self.n_wxr,rotation=self.rotation,ansatz=self.ansatz,n_parallel=self.n_parallel_x,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
				x_layer.w_a = self.wxa
				x_layer.w_r = self.wxr
				h_layer = AnsatzRotationLinear(h_0.shape[0],self.n_hidden,n_weights_a=self.n_wha,n_weights_r=self.n_whr,rotation=self.rotation,ansatz=self.ansatz,n_parallel=self.n_parallel_h,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
				h_layer.w_a = self.wha
				h_layer.w_r = self.whr
				out1 = x_layer(x_t)
				out2 = h_layer(h_0.reshape(1,h_0.shape[0]))
				h_temp = (out1 + out2)/2
				h[sample,t,:] = h_temp.flatten()
				h_0 = h_temp.flatten()
				if not self.seed_simulator is None:
					self.seed_simulator+=1
		return(h)


class IntermediateAnsatzRotationLinear:
	def __init__(self,n_qubits=None,n_outputs=None,n_weights_a=None,n_weights_r=None,ansatz_i=None,ansatz_a=None,rotation=None,n_parallel = 1,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,classical_bits=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_qubits= n_qubits
		self.n_outputs = n_outputs
		self.w_r = np.random.randn(n_outputs,n_weights_r)
		self.n_parallel = n_parallel
		self.n_weights_a = n_weights_a
		self.n_weights_r = n_weights_r
		if n_parallel > 1:
			self.w_a = np.random.randn(1,n_weights_a)
			self.w_size = n_outputs*n_weights_r + n_weights_a
		else:
			self.w_a = np.random.randn(n_outputs,n_weights_a)
			self.w_size = n_outputs*n_weights_a + n_weights_r*n_outputs
		self.ansatz_a=ansatz_a
		self.ansatz_i = ansatz_i
		self.rotation=rotation
		self.shots=shots
		self.seed_simulator = seed_simulator
		self.backend = backend
		self.noise_model = noise_model
		self.basis_gates=basis_gates
		self.classical_bits = classical_bits
		self.coupling_map=coupling_map
		self.seed_transpiler=seed_transpiler
		self.transpile=transpile
		self.optimization_level=optimization_level
		self.error_mitigator = error_mitigator

	def set_weights(self,w,w_idx=0):
		if self.n_parallel > 1:
			self.w_a = w[w_idx:(w_idx+self.n_weights_a)].copy().reshape(1,self.n_weights_a)
			w_idx += self.n_weights_a
		else:
			self.w_a = w[w_idx:(w_idx+self.n_outputs*self.n_weights_a)].copy().reshape(self.n_outputs,self.n_weights_a)
			w_idx += self.n_outputs*self.n_weights_a
		self.w_r = w[w_idx:(w_idx + self.n_outputs*self.n_weights_r)].copy().reshape(self.n_outputs,self.n_weights_r)
		w_idx += self.n_outputs*self.n_weights_r
		return(w_idx)

	def __call__(self,x):
		output = np.zeros((x.shape[0],self.n_outputs))
		for sample in range(x.shape[0]):
			for i in range(0,self.n_outputs,self.n_parallel):
				if (self.n_outputs - i) < self.n_parallel:
					n_parallel = self.n_outputs - i
				else:
					n_parallel = self.n_parallel
				rotation_register = qk.QuantumRegister(self.n_qubits)
				if not self.classical_bits is None:
					classical_register = qk.ClassicalRegister(self.classical_bits)
				else:
					classical_register = qk.ClassicalRegister(n_parallel)
				ancilla_register = qk.QuantumRegister(n_parallel)
				circuit = qk.QuantumCircuit(rotation_register,ancilla_register,classical_register)
				registers = [rotation_register,ancilla_register,classical_register]
				circuit,registers = self.ansatz_i(2*np.pi*x[sample,:],circuit,registers)
				if self.n_parallel > 1:
					circuit,registers = self.ansatz_a(self.w_a[0,:],circuit,registers)
				else:
					circuit,registers = self.ansatz_a(self.w_a[i,:],circuit,registers)
				for j in range(n_parallel):
					circuit,registers = self.rotation(self.w_r[i+j,:],j,circuit,registers)
				if not self.classical_bits is None and (self.n_parallel == self.n_outputs):
					return(circuit,registers)
				circuit.measure(registers[1],registers[-1])
				if self.transpile:
					circuit = qk.compiler.transpile(circuit,backend=self.backend,backend_properties=self.backend.properties(),seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,basis_gates=self.basis_gates,coupling_map=self.coupling_map)
				job = qk.execute(circuit, backend = self.backend,seed_simulator=self.seed_simulator,shots=self.shots,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map).result()
				if not self.error_mitigator is None:
					n_qubits = circuit.n_qubits
					meas_filter = self.error_mitigator(n_qubits,list(range(n_qubits-n_parallel,n_qubits)),self.backend,seed_simulator=self.seed_simulator,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,shots=self.shots)
					result = meas_filter.apply(job)
					result = result.get_counts(0)
				else:
					result = job.get_counts(circuit)
				out = np.zeros(n_parallel)
				for key,value in result.items():
					key_ = key[::-1]
					for k,qubit in enumerate(key_):
						if qubit == '1':
							out[k] += value
				out /= self.shots
				output[sample,i:(i+n_parallel)] = out
				if not self.seed_simulator is None:
					self.seed_simulator+=1
		return(output)



class GeneralLinear:
	def __init__(self,n_qubits=None,n_outputs=None,n_weights_a=None,n_weights_ent=None,U_enc=None,U_a=None,U_ent=None,n_parallel = 1,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,classical_bits=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_qubits= n_qubits
		self.n_outputs = n_outputs
		self.w_r = np.random.randn(n_outputs,n_weights_ent)
		self.n_parallel = n_parallel
		self.n_weights_a = n_weights_a
		self.n_weights_ent = n_weights_ent
		if n_parallel > 1:
			self.w_a = np.random.randn(1,n_weights_a)
			self.w_size = n_outputs*n_weights_ent + n_weights_a
		else:
			self.w_a = np.random.randn(n_outputs,n_weights_a)
			self.w_size = n_outputs*n_weights_a + n_weights_ent*n_outputs
		self.U_ent = U_ent
		self.U_enc = U_enc
		self.U_a = U_a
		self.shots=shots
		self.seed_simulator = seed_simulator
		self.backend = backend
		self.noise_model = noise_model
		self.basis_gates=basis_gates
		self.classical_bits = classical_bits
		self.coupling_map=coupling_map
		self.seed_transpiler=seed_transpiler
		self.transpile=transpile
		self.optimization_level=optimization_level
		self.error_mitigator = error_mitigator

	def set_weights(self,w,w_idx=0):
		if self.n_parallel > 1:
			self.w_a = w[w_idx:(w_idx+self.n_weights_a)].copy().reshape(1,self.n_weights_a)
			w_idx += self.n_weights_a
		else:
			self.w_a = w[w_idx:(w_idx+self.n_outputs*self.n_weights_a)].copy().reshape(self.n_outputs,self.n_weights_a)
			w_idx += self.n_outputs*self.n_weights_a
		self.w_r = w[w_idx:(w_idx + self.n_outputs*self.n_weights_ent)].copy().reshape(self.n_outputs,self.n_weights_ent)
		w_idx += self.n_outputs*self.n_weights_ent
		return(w_idx)

	def __call__(self,x):
		output = np.zeros((x.shape[0],self.n_outputs))
		for sample in range(x.shape[0]):
			for i in range(0,self.n_outputs,self.n_parallel):
				enc = deepcopy(self.U_enc)
				ansatz_a = deepcopy(self.U_a)
				rotation = deepcopy(self.U_ent)
				if self.n_parallel > 1:
					if (self.n_outputs - i) < self.n_parallel:
						n_parallel = self.n_outputs - i
					else:
						n_parallel = self.n_parallel
				else:
					n_parallel = self.n_parallel
				rotation_register = qk.QuantumRegister(self.n_qubits)
				if not self.classical_bits is None:
					classical_register = qk.ClassicalRegister(self.classical_bits)
				else:
					classical_register = qk.ClassicalRegister(n_parallel)
				circuit = qk.QuantumCircuit(rotation_register,classical_register)
				registers = [rotation_register,classical_register]
				circuit,registers = enc(2*np.pi*x[sample,:],circuit,registers)
				ancilla_register = qk.QuantumRegister(n_parallel)
				circuit.add_register(ancilla_register)
				registers.insert(1,ancilla_register)
				if self.n_parallel > 1:
					circuit,registers = ansatz_a(self.w_a[0,:],circuit,registers)
				else:
					circuit,registers = ansatz_a(self.w_a[i,:],circuit,registers)
				for j in range(n_parallel):
					circuit,registers = rotation(self.w_r[i+j,:],j,circuit,registers)
				if not self.classical_bits is None and (self.n_parallel == self.n_outputs):
					return(circuit,registers)
				circuit.measure(registers[1],registers[-1])
				if self.transpile:
					circuit = qk.compiler.transpile(circuit,backend=self.backend,backend_properties=self.backend.properties(),seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,basis_gates=self.basis_gates,coupling_map=self.coupling_map)
				job = qk.execute(circuit, backend = self.backend,seed_simulator=self.seed_simulator,shots=self.shots,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map).result()
				if not self.error_mitigator is None:
					n_qubits = circuit.n_qubits
					meas_filter = self.error_mitigator(n_qubits,list(range(n_qubits-n_parallel,n_qubits)),self.backend,seed_simulator=self.seed_simulator,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,shots=self.shots)
					result = meas_filter.apply(job)
					result = result.get_counts(0)
				else:
					result = job.get_counts(circuit)
				out = np.zeros(n_parallel)
				for key,value in result.items():
					key_ = key[::-1]
					for k,qubit in enumerate(key_):
						if qubit == '1':
							out[k] += value
				out /= self.shots
				output[sample,i:(i+n_parallel)] = out
				if not self.seed_simulator is None:
					self.seed_simulator+=1
		return(output)