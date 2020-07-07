import qiskit as qk 
import numpy as np
from utils import *


def ansatz(theta,circuit,registers):
	return(circuit,registers)

n = 16

def initial_state(circuit,registers):
	for i in range(0,0):
		circuit.x(registers[0][i])
	return(circuit,registers)





def expectation_value(hamiltonian_list,n_qubits):
		"""
		Calculates expectation values and adds them together
		"""
		E = 0
		for pauli_string in hamiltonian_list:
			factor = pauli_string[0]
			if factor == 0:
				continue
			classical_bits = len(pauli_string[1:])
			if classical_bits == 0:
				E += factor
				continue
			circuit,registers = initialize_circuit(n_qubits,0,classical_bits)
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = ansatz(0,circuit,registers)
			qubit_list = []
			for qubit,gate in pauli_string[1:]:
				circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
				qubit_list.append(qubit)
			E += measure_expectation_value(qubit_list,factor,circuit,registers)
		print('<E> = ', E)
		return(E)




print(expectation_value(pair_number_operator(n),n))
