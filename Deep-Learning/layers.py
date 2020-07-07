import numpy as np
import sys
sys.path.append('../')
from utils import *
from dl_utils import *



class GeneralLinear:
	def __init__(self,n_qubits=None,n_outputs=None,n_weights_a=None,n_weights_ent=None,U_enc=None,U_a=None,U_ent=None,bias=False,n_parallel = 1,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,classical_bits=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_qubits= n_qubits
		self.n_outputs = n_outputs
		self.w_r = np.random.randn(n_outputs,n_weights_ent)
		self.n_parallel = n_parallel
		self.n_weights_a = n_weights_a if not bias else (n_weights_a + 1)
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
		self.bias = bias

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
			if self.bias:
				x_ = np.zeros(x.shape[1] if not self.bias else x.shape[1]+1)
				x_[0] = 1/np.sqrt(2)
				x_[1:] = x[sample,:].copy()/(np.sqrt(np.sum(x[sample,:]**2))*np.sqrt(2) + 1e-14)
			else:
				x_ = x[sample,:].copy()
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
				circuit,registers = enc(2*np.pi*x_,circuit,registers)
				if self.n_parallel > 1:
					circuit,registers = ansatz_a(self.w_a[0,:],circuit,registers)
				else:
					circuit,registers = ansatz_a(self.w_a[i,:],circuit,registers)
				ancilla_register = qk.QuantumRegister(n_parallel)
				circuit.add_register(ancilla_register)
				registers.insert(1,ancilla_register)
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


class GeneralRecurrent:
	def __init__(self,n_qubits=None,n_hidden=None,n_weights_a=None,n_weights_ent=None,U_enc=None,U_a=None,U_ent=None,bias=False,n_parallel=1,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,coupling_map=None,transpile=False,seed_transpiler=None,optimization_level=1,error_mitigator=None):
		self.shots = shots
		self.n_hidden = n_hidden
		self.n_outputs = n_hidden
		self.wxr = np.random.randn(n_hidden,n_weights_ent)
		self.n_weights_a = n_weights_a if not bias else (n_weights_a + 1)
		self.n_weights_ent = n_weights_ent
		self.w_size = self.n_hidden*n_weights_ent
		if n_parallel > 1:
			self.wxa = np.random.randn(1,n_weights_a)
			self.w_size += n_weights_a
		else:
			self.wxa = np.random.randn(n_hidden,n_weights_a)
			self.w_size += self.n_outputs*n_weights_a
		self.U_enc=U_enc
		self.U_a = U_a
		self.U_ent = U_ent
		self.shots=shots
		self.n_parallel = n_parallel
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.noise_model = noise_model
		self.basis_gates=basis_gates
		self.coupling_map=coupling_map
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.error_mitigator=None
		self.bias = bias
		self.n_qubits = n_qubits

	
	def set_weights(self,w,w_idx=0):
		if self.n_parallel > 1:
			self.wxa = w[w_idx:(w_idx+self.n_weights_a)].copy().reshape(1,self.n_weights_a)
			w_idx += self.n_weights_a
		else:
			self.wxa = w[w_idx:(w_idx+self.n_outputs*self.n_weights_a)].copy().reshape(self.n_outputs,self.n_weights_a)
			w_idx += self.n_outputs*self.n_weights_a_x
		self.wxr = w[w_idx:(w_idx + self.n_outputs*self.n_weights_ent)].copy().reshape(self.n_outputs,self.n_weights_ent)
		w_idx += self.n_outputs*self.n_weights_ent
		return(w_idx)

	def __call__(self,x,h_0 = None):
		if h_0 is None:
			h_0 = np.ones(self.n_hidden)
		timesteps = x.shape[1]
		h = np.zeros((x.shape[0],timesteps,h_0.shape[0]))
		for sample in range(x.shape[0]):
			for t in range(x.shape[1]):
				xh = np.concatenate((x[sample,t,:],h_0))
				xh = xh.reshape(1,xh.shape[0])
				layer = GeneralLinear(self.n_qubits,self.n_hidden,n_weights_a=self.n_weights_a if not self.bias else self.n_weights_a - 1,n_weights_ent=self.n_weights_ent,U_enc=self.U_enc,U_ent=self.U_ent,U_a=self.U_a,bias=self.bias,n_parallel=self.n_parallel,seed_simulator=self.seed_simulator,backend=self.backend,noise_model=self.noise_model,basis_gates=self.basis_gates,coupling_map=self.coupling_map,transpile=self.transpile,seed_transpiler=self.seed_transpiler,optimization_level=self.optimization_level,error_mitigator=self.error_mitigator)
				layer.w_a = self.wxa
				layer.w_r = self.wxr
				h_temp = layer(xh)
				h[sample,t,:] = h_temp.flatten()
				h_0 = h_temp.flatten()
				if not self.seed_simulator is None:
					self.seed_simulator+=1
		return(h)