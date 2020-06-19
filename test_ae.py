from utils import ControlledTimeEvolutionOperator
import qiskit as qk 

hamiltonian_list = [[3,[0,'x']],[-2,[5,'y'],[3,'z']],[2]]
U = ControlledTimeEvolutionOperator(hamiltonian_list,dt=0.001,T=1)

control_register = qk.QuantumRegister(2)
evo_register = qk.QuantumRegister(6)
ancilla_register = qk.QuantumRegister(1)
classical_register = qk.QuantumRegister(4)
circuit = qk.QuantumCircuit(control_register,evo_register,ancilla_register,classical_register)
registers = [control_register,evo_register,ancilla_register,classical_register]

circuit,registers = U(circuit=circuit,registers=registers,control=1,power=2**6)
