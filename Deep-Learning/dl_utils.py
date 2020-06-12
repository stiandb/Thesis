import numpy as np


class Utils:
	def set_weights(self,w):
		w_idx = 0
		w = w.flatten()
		for layer in self.layers:
			w_idx = layer.set_weights(w,w_idx)


def y_rotation(weights,ancilla,circuit,registers):
	n = len(registers[0])
	for i in range(n):
		circuit.cry(weights[i],registers[0][i],registers[1][ancilla])
	return(circuit,registers)

def euler_rotation(weights,ancilla,circuit,registers):
	i = 0
	n = len(registers[0])
	for q in range(n):
		circuit.crz(weights[i],registers[0][q],registers[1][ancilla])
		circuit.crx(weights[i+1],registers[0][q],registers[1][ancilla])
		circuit.crz(weights[i+2],registers[0][q],registers[1][ancilla])
		i+=3
	return(circuit,registers)
	