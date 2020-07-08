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

