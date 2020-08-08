import numpy as np


class Utils:
	"""
	Some utilities to be utilized by neural networks
	"""
	def set_weights(self,w):
	"""
	Sets the weights to w for neural network constructed with class QDNN
	Input:
		w (numpy 1d array) - Array of all weights for neural network
	"""
		w_idx = 0
		w = w.flatten()
		for layer in self.layers:
			if type(layer) is list:
				for sub_layer in layer:
					w_idx = sub_layer.set_weights(w,w_idx)
			else:
				w_idx = layer.set_weights(w,w_idx)


class YRotation:
	"""
	Performs y-rotation conditioned on encoded register to an ancilla register
	"""
	def __init__(self,bias=False):
		"""
		Input:
			bias (boolean) - Applies non-conditional rotation (bias) to ancilla qubit if set to True
		"""
		self.bias = bias
	def __call__(self,weights,ancilla,circuit,registers):
		"""
		Input:
			weights (numpy 1d array) - Weights for ansatz
			ancilla (int) - Index of ancilla qubit to apply conditional rotation to
			circuit (qiskit QuantumCircuit) - circuit for neural network
			registers (list) - List containing encoded register as first element, while 
								the second element is the ancilla register
		Output:
			circuit (qiskit QuantumCircuit) - Circuit with applied entangler on
			registers (list) - List containing corresponding registers
		"""
		if self.bias:
			circuit.ry(weights[-1],registers[1][ancilla])
		n = len(registers[0])
		for i in range(n):
			circuit.cry(weights[i],registers[0][i],registers[1][ancilla])
		return(circuit,registers)



class EulerRotation:
	"""
	Performs Euler-rotation conditioned on encoded register to an ancilla register
	"""
	def __init__(self,bias=False):
		"""
		Input:
			bias (boolean) - Applies non-conditional rotation (bias) to ancilla qubit if set to True
		"""
		self.bias = bias
	def __call__(self,weights,ancilla,circuit,registers):
		"""
		Input:
			weights (numpy 1d array) - Weights for ansatz
			ancilla (int) - Index of ancilla qubit to apply conditional rotation to
			circuit (qiskit QuantumCircuit) - circuit for neural network
			registers (list) - List containing encoded register as first element, while 
								the second element is the ancilla register
		Output:
			circuit (qiskit QuantumCircuit) - Circuit with applied entangler on
			registers (list) - List containing corresponding registers
		"""
		i = 0
		n = len(registers[0])
		if self.bias:
			circuit.ry(weights[-1],registers[1][ancilla])
		for q in range(n):
			circuit.crz(weights[i],registers[0][q],registers[1][ancilla])
			circuit.mcrx(weights[i+1],[registers[0][q]],registers[1][ancilla])
			circuit.crz(weights[i+2],registers[0][q],registers[1][ancilla])
			i+=3
		return(circuit,registers)

class EntanglementRotation:
	"""
	Flips ancilla qubit if all encoded qubits are in the zero or one-state
	"""
	def __init__(self,bias=False,zero_condition=False):
		"""
		Input:
			bias (boolean) - If True, a rotation (bias) is applied to the ancilla qubit
			zero_condition (boolean) - If True, the flip of the ancilla is conditioned on the
										encoded qubits being in the zero state. Else, it is conditioned
										on the encoded qubits being in the one state.
		"""
		self.bias=bias
		self.zero_condition=zero_condition
	def __call__(self,weights,ancilla,circuit,registers):
		"""
		Input:
			weights (numpy 1d array) - Weights for ansatz
			ancilla (int) - Index of ancilla qubit to apply conditional rotation to
			circuit (qiskit QuantumCircuit) - circuit for neural network
			registers (list) - List containing encoded register as first element, while 
								the second element is the ancilla register
		Output:
			circuit (qiskit QuantumCircuit) - Circuit with applied entangler on
			registers (list) - List containing corresponding registers
		"""
		if self.zero_condition:
			for i in range(len(registers[0])):
				circuit.x(registers[0][i])
		if self.bias:
			circuit.ry(weights[0],registers[1][ancilla])
		circuit.mcrx(np.pi,[registers[0][i] for i in range(len(registers[0]))],registers[1][ancilla])
		return(circuit,registers)

