from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.circuit import Parameter
from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere





def make_circuit(circ, param_1, where ):
 
 theta, zeta, chi, gamma, phi=param_1
 q0, q1=where
 #print (where, q0, q1, theta, gamma, phi)


 circ.rz(-2.*phi,q0)

 circ.rz(chi+zeta,q1)


 circ.h(q0)
 circ.cz(q0, q1)
 circ.h(q0)

 circ.rz(-2.*gamma,q0)
 circ.ry(-theta,q1)


 circ.h(q1)
 circ.cz(q0, q1)
 circ.h(q1)

 circ.ry(theta,q1)

 circ.h(q0)
 circ.cz(q0, q1)
 circ.h(q0)

 circ.rz(-zeta,q0)
 circ.rz(-chi,q1)


 return circ








list_params=load_from_disk("list_params")
print (list_params)

list_qubits=load_from_disk("list_qubits")
print (list_qubits)

L=6
circ = QuantumCircuit(L,L)

#Register qubit
for i in range(L):
 if i%2!=0:  
  circ.x(i)



for i in range(len(list_params)):
     param_1=list_params[i]
     where=list_qubits[i]
     circ=make_circuit( circ, param_1, where )




circ.draw(output='mpl', filename='my_circuit.pdf')
#plt.savefig('circ.pdf')
#plt.clf()
backend = Aer.get_backend('statevector_simulator')
job = execute(circ, backend)
result_sim = job.result()
psi  = result_sim.get_statevector(circ, decimals=5)

print ( "psi", psi, "\n", "E=", psi.conj().T @ ham_heis(L) @ psi, psi.conj().T @ psi
)
plot_state_city(psi)
plt.savefig('psi-city.pdf')
plt.clf()

plot_state_hinton(psi)
plt.savefig('psi-hinton.pdf')
plt.clf()


plot_state_paulivec(psi, title="My Paulivec", color=['purple', 'orange', 'green'])
plt.savefig('psi-paulivec.pdf')
plt.clf()



# backend = Aer.get_backend('unitary_simulator')
# job = execute(circ, backend)
# result_sim = job.result()
# U  = result_sim.get_unitary(circ, decimals=5)
# print ("U", U, type(U), np.shape(U))







meas = QuantumCircuit(L, L)
meas.barrier(range(L))
meas.measure(range(L), range(L))
qc = circ + meas
qc.draw(output='mpl', filename='finalCirc.pdf')

backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend, shots=1000)
result_sim = job.result()





counts = result_sim.get_counts(qc)
print(counts)

