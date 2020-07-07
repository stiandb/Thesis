import qiskit as qk 
import numpy as np
from utils import inference_inner_product


def U_a(theta,circuit,registers):
	"""circuit.cx(registers[-2][0],registers[0][0])
				circuit.cx(registers[-2][0],registers[0][1])
				circuit.cx(registers[-2][0],registers[0][2])
				circuit.cx(registers[-2][0],registers[0][3])"""
	return(circuit,registers)


def initial_state(circuit,registers):
	for i in range(int(len(registers[0])/2),len(registers[0])):
		circuit.cx(registers[-2][0],registers[0][i])
	return(circuit,registers)


def initialize_circuit(n):
	qr = qk.QuantumRegister(n)
	cr = qk.ClassicalRegister(1)
	circuit = qk.QuantumCircuit(qr,cr)
	registers = [qr,cr]
	return(circuit,registers)
n=4
inner_product = 0
for j in range(1,n,2):
	for k in range(1,n,2):
		if k == j:
			continue
		circuit,registers = initialize_circuit(n)

		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)

		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)

		circuit,registers = initialize_circuit(n)
		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
		
		def cU_b(circuit,registers):	
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k])
			return(circuit,registers)
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)

		circuit,registers = initialize_circuit(n)
		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k-1])
			return(circuit,registers)

		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)


		circuit,registers = initialize_circuit(n)

		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)

		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k-1])
			circuit.cz(registers[-2][0],registers[0][k])
			return(circuit,registers)
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)

		circuit,registers = initialize_circuit(n)

		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j-1])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)

		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)

		circuit,registers = initialize_circuit(n)

		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j-1])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)

		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k])
			return(circuit,registers)
			
			
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)
		circuit,registers = initialize_circuit(n)
		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j-1])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)

		def cU_b(circuit,registers):
			
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k-1])
			return(circuit,registers)
			
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)

		circuit,registers = initialize_circuit(n)

		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j-1])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
		
		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k-1])
			circuit.cz(registers[-2][0],registers[0][k])
			return(circuit,registers)
			
			
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)

		circuit,registers = initialize_circuit(n)
		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j-1])
			circuit.cz(registers[-2][0],registers[0][j])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)

		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
			
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)
		circuit,registers = initialize_circuit(n)
		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j-1])
			circuit.cz(registers[-2][0],registers[0][j])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k])
			return(circuit,registers)
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)


		circuit,registers = initialize_circuit(n)

		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j-1])
			circuit.cz(registers[-2][0],registers[0][j])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k-1])
			return(circuit,registers)
		
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)
		circuit,registers = initialize_circuit(n)

		def cU_a(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit.cz(registers[-2][0],registers[0][j-1])
			circuit.cz(registers[-2][0],registers[0][j])
			circuit,registers = U_a(0,circuit,registers)
			return(circuit,registers)
		def cU_b(circuit,registers):
			circuit,registers = initial_state(circuit,registers)
			circuit,registers = U_a(0,circuit,registers)
			circuit.cz(registers[-2][0],registers[0][k-1])
			circuit.cz(registers[-2][0],registers[0][k])
			return(circuit,registers)
			
		inner_product += inference_inner_product(cU_a,cU_b,circuit,registers)
print(inner_product)


