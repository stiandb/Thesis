import numpy as np


class Utils:
	def set_weights(self,w):
		w_idx = 0
		w = w.flatten()
		for layer in self.layers:
			w_idx = layer.set_weights(w,w_idx)


class YRotation:
	def __init__(self,bias=False):
		self.bias = bias
	def __call__(self,weights,ancilla,circuit,registers):
		if self.bias:
			circuit.ry(weights[-1],registers[1][ancilla])
		n = len(registers[0])
		for i in range(n):
			circuit.cry(weights[i],registers[0][i],registers[1][ancilla])
		return(circuit,registers)


class EulerRotation:
	def __init__(self,bias=False):
		self.bias = bias
	def __call__(self,weights,ancilla,circuit,registers):
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