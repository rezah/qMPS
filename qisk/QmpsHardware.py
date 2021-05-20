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
import quf

Gate="SU4"
Gate="FSIMG"

list_params_U=load_from_disk(f"list_params_U{Gate}")
list_qubits_U=load_from_disk(f"list_qubits_U{Gate}")
list_params=load_from_disk(f"list_params{Gate}")
list_qubits=load_from_disk(f"list_qubits{Gate}")
n_ancilla=load_from_disk( f"Qbit_rho")
list_basis=load_from_disk( f"list_basis_cirq")


list_tag_block=load_from_disk("list_tag_block")
#print (list_tag_block[0], len(list_tag_block[0]))
#physical + bond qubits
#l+number of qubit+ ancilla
n_Qbit=4
l=8
L=n_Qbit+n_ancilla+l
U=4.0
t=1.0
mu=0

#Register qubit


circ = QuantumCircuit(L)
circ_temp = QuantumCircuit(L)
count_val=0
if Gate=="FSIMG":
 for i in range(len(list_basis)):
  if list_basis[i]=="1":
   #print (i, list_basis[i], list_basis)
   circ.x(i)
   circ_temp.x(i)



for i in range(len(list_params_U)):
     param_1=list_params_U[i]
     where=list_qubits_U[i]
     #print ("F",where)
     if Gate=="FSIMG":
         circ=quf.make_circuit( circ, param_1, where )
         circ_temp=quf.make_circuit(circ_temp, param_1, where )
     else:
         circ=quf.make_circuit_gen(circ, param_1, where )
         circ_temp=quf.make_circuit_gen(circ_temp, param_1, where )

circ.barrier(range(L))
circ_temp.barrier(range(L))
circ_temp.draw(output='mpl', filename='Figs/circuit_rho.pdf')
circ_temp = QuantumCircuit(L)




for i in range(len(list_params)):
     param_1=list_params[i]
     where=list_qubits[i]
     t0, t1=where
     if t0 >= n_Qbit:
       t0=t0+n_ancilla
     if t1 >= n_Qbit:
       t1=t1+n_ancilla
     where=t0, t1
     #print ("S",where)
     if Gate=="FSIMG":
         circ=quf.make_circuit( circ, param_1, where )
         circ_temp=quf.make_circuit(circ_temp, param_1, where )
     else:
         circ=quf.make_circuit_gen(circ, param_1, where )
         circ_temp=quf.make_circuit_gen(circ_temp, param_1, where )
     if (i+1)%(len(list_tag_block[0]))==0:
      #print ("i", i)
      circ.barrier(range(L))
      circ_temp.barrier(range(L))
      circ_temp.draw(output='mpl', filename=f'Figs/circuit{count_val}.pdf')
      count_val+=1
      circ_temp = QuantumCircuit(L)






circ.draw(output='mpl', filename='my_circuit.pdf')
plt.savefig('circ.pdf')
plt.clf()
backend = Aer.get_backend('statevector_simulator')
job = execute(circ, backend)
result_sim = job.result()
psi  = result_sim.get_statevector(circ, decimals=5)
# 
# 
# #print ("psi", psi)
# MPO_origin=quf.mpo_Fermi_Hubburd(L//2, U, t, mu)
# MPO_N=quf.mpo_particle(L//2)
# MPO_up, MPO_down=quf.mpo_spin(L//2)
# #print ("E_exact", -6.3474)
# #print (  "E=", psi.conj().T @ MPO_origin.to_dense() @ psi)
# #print (  "N=", psi.conj().T @ MPO_N.to_dense() @ psi)
# #print (  "Up=", psi.conj().T @ MPO_up.to_dense() @ psi)
# #print (  "Down=", psi.conj().T @ MPO_down.to_dense() @ psi)
# 
# MPO_I=MPO_identity(L, phys_dim=2)
# MPO_result=MPO_identity(L, phys_dim=2)
# MPO_result=MPO_result*0.0
# MPO_f=MPO_result*0.0
# max_bond_val=200
# cutoff_val=1.0e-12
# 
# 
for i in range( L):

 MPO_I=MPO_identity(L, phys_dim=2)
 W = np.zeros([ 1, 1, 2, 2])
 Z = qu.pauli('Z')
 X = qu.pauli('X')
 Y = qu.pauli('Y')
 I = qu.pauli('I')
 S_up=(X+1.0j*Y)*(0.5)
 S_down=(X-1.0j*Y)*(0.5)
 Wl = np.zeros([ 1, 2, 2], dtype='float64')
 W = np.zeros([1, 1, 2, 2], dtype='float64')
 Wr = np.zeros([ 1, 2, 2], dtype='float64')
 
 Wl[ 0,:,:]=S_up@S_down
 W[ 0,0,:,:]=S_up@S_down
 Wr[ 0,:,:]=S_up@S_down


 W_list=[Wl]+[W]*(L-2)+[Wr]

 MPO_I[i].modify(data=W_list[i])
 E_final=psi.conj().T @ MPO_I.to_dense() @ psi  
 print ("i", i, "X", E_final.real )

# E_u=0
# for i in range(2):
#   MPO_I=MPO_identity(L, phys_dim=2)
#   MPO_I[2*i].modify(data=W_list[2*i])
#   MPO_I[2*i+1].modify(data=W_list[2*i+1])
#   MPO_I=MPO_I*U
#   E_final2=psi.conj().T @ MPO_I.to_dense() @ psi  
#   print ("i", i, "U", E_final2.real )
#   E_u+=E_final2.real
# 
# 
# E_t=0
# for i in range(2):
#   Wl = np.zeros([ 1, 2, 2], dtype='float64')
#   W = np.zeros([1, 1, 2, 2], dtype='float64')
#   Wr = np.zeros([ 1, 2, 2], dtype='float64')
# 
#   Wl[ 0,:,:]=S_up
#   W[ 0,0,:,:]=S_up
#   Wr[ 0,:,:]=S_up
#   W_1=[Wl]+[W]*(L-2)+[Wr]
# 
#   Wl = np.zeros([ 1, 2, 2], dtype='float64')
#   W = np.zeros([1, 1, 2, 2], dtype='float64')
#   Wr = np.zeros([ 1, 2, 2], dtype='float64')
# 
#   Wl[ 0,:,:]=S_down
#   W[ 0,0,:,:]=S_down
#   Wr[ 0,:,:]=S_down
#   W_2=[Wl]+[W]*(L-2)+[Wr]
# 
#   MPO_I=MPO_identity(L, phys_dim=2 )
#   MPO_I[2*i].modify(data=W_1[2*i])
#   MPO_I[2*i+2].modify(data=W_2[2*i+2])
#   MPO_result=MPO_I*1.0
#   MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
# 
#   MPO_I=MPO_identity(L, phys_dim=2 )
#   MPO_I[2*i].modify(data=W_2[2*i])
#   MPO_I[2*i+2].modify(data=W_1[2*i+2])
#   MPO_result=MPO_result+MPO_I
#   MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
# 
# 
#   MPO_I=MPO_identity(L, phys_dim=2 )
#   MPO_I[2*i+1].modify(data=W_1[2*i+1])
#   MPO_I[2*i+3].modify(data=W_2[2*i+3])
#   MPO_result=MPO_result+MPO_I
#   MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
# 
#   MPO_I=MPO_identity(L, phys_dim=2 )
#   MPO_I[2*i+1].modify(data=W_2[2*i+1])
#   MPO_I[2*i+3].modify(data=W_1[2*i+3])
#   MPO_result=MPO_result+MPO_I
#   MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
#   MPO_f=MPO_result*(t)
#   MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )
# 
#   E_final1=psi.conj().T @ MPO_f.to_dense() @ psi  
#   print ("i", i, "t", E_final1.real )
#   E_t+=E_final1.real
# 
# 
# print ("final", E_t,E_u, (E_u+E_t)/2.  )









