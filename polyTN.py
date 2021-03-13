from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import quf
import matplotlib.pyplot as plt
from numpy import linalg as LA
import string
J=1.0
h=1.0
seed_val=2
d=2
D=2
L=24
N_lay=4




list_mpo=[]
for i in range(N_lay):
  if i==0:
   mpo=MPS_rand_state(L=L, bond_dim=D, phys_dim=D,  normalize=True)
   mpo.add_tag(f"G{i}", where=None, which='all')
  else:
   mpo = MPO_rand(L, bond_dim=D, phys_dim=D)
   mpo.add_tag(f"G{i}", where=None, which='all')
  list_mpo.append(mpo)





list_ids=[]
list_alphabet=list(string.ascii_lowercase)
for i in range(N_lay):
 list_ids.append(f"__ind_{list_alphabet[i]}{{}}__")

list_tn=align_TN_1D(*list_mpo, ind_ids={*list_ids}, inplace=False)


TN_l=qtn.TensorNetwork(list_tn)
  
rotate_ten_list=[]
for i in range(L):
 rotate_ten_list.append(qtn.Tensor(qu.rand(D*d).reshape(D, d), inds=(f'b{i}',f'k{i}'), tags={"R"}))

R_l=qtn.TensorNetwork(rotate_ten_list)

TN_l=TN_l & R_l


for i in range(L):
 TN_l.contract_ind(f'b{i}', optimize='auto-hq')

list_tags=[]
for i in range(N_lay):
  list_tags.append(f"G{i}")


#print (TN_l.graph(color=list_tags,show_inds=all, show_tags=False, iterations=4800, k=None, fix=None, figsize=(30, 30),node_size=200) )
#print (TN_l)






ham = MPO_ham_heis(L, j=1.0, bz=0.0, S=0.5, cyclic=False)
dmrg = DMRG2(ham, bond_dims=[10, 20, 60, 80, 100, 200, 200, 300], cutoffs=1e-10)
dmrg.solve(tol=1e-8, verbosity=0 )
print("DMRG=", dmrg.energy)
E_exact=dmrg.energy
p_DMRG=dmrg.state
E_exact
list_A=[1,2,3,4,5,4]
J=[1,2,3,5]





def norm_TN(TN_l):
  noem_val=(TN_l.H & TN_l).contract(all, optimize='auto-hq')  
  TN_l=TN_l*(noem_val**(-0.5))
  return TN_l

def re_ind_k(TN_l, L):
    for i in range(L): 
     t=list((TN_l[[f"G{N_lay-1}", f"I{i}"]].inds))
     for m in range(len(t)):
      if t[m]==f'b{i}':
          t[m]=f'k{i}'
     TN_l[[f"G{N_lay-1}", f"I{i}"]].modify(inds=t)            
    return  TN_l

def re_ind_b(TN_l, L):
    for i in range(L): 
     t=list((TN_l[[f"G{N_lay-1}", f"I{i}"]].inds))
     for m in range(len(t)):
      if t[m]==f'k{i}':
          t[m]=f'b{i}'
     TN_l[[f"G{N_lay-1}", f"I{i}"]].modify(inds=t)            
    return  TN_l

    
def energy_f(TN_l, ham):
   TN_l_h=TN_l.H  
   TN_l_h=re_ind_b(TN_l_h, L)        

   return ( TN_l_h & ham & TN_l).contract(all, optimize='auto-hq')

def loss_f(TN_l, p_DMRG):
    return  1-abs((TN_l & p_DMRG.H).contract(all, optimize='auto-hq'))





TN_l=norm_TN(TN_l)
TN_l=re_ind_k(TN_l, L)        


TN_l_h=TN_l.H  

TN_l_h=re_ind_b(TN_l_h, L)        
#print ( (TN_l_h & ham & TN_l).graph(color=["G0", "G1", "G2", "G3"]), energy_f(TN_l, ham))
tnopt = qtn.TNOptimizer(
    TN_l,                          # the initial TN
    loss_fn=energy_f,                         # the loss function
    norm_fn=norm_TN,                         # this is the function that 'prepares'/constrains the tn
    constant_tags=[],                   # only optimize the tensors tagged 'G'
    loss_constants={'ham': ham },  # additional tensor/tn kwargs
    loss_kwargs={},
    tags=[],
    optimizer='L-BFGS-B',             # how to optimize
    autodiff_backend='torch',       # how to generate/compute the gradient   
    device='cpu', 
)
#TN_l = tnopt.optimize(10, tol=1e-12, ftol=1e-12, gtol=1e-12)




plt.plot(tnopt.losses, color='green')


#print ( (TN_l & p_DMRG).graph(color=["G0", "G1", "G2", "G3"],show_inds=all))

#print ( (p_DMRG & TN_l)^all, TN_l, p_DMRG)

tnopt_psi = qtn.TNOptimizer(
    TN_l,                      
    loss_fn=loss_f,                    
    norm_fn=norm_TN,
    loss_constants={'p_DMRG': p_DMRG},
    constant_tags=[],    
    tags=[*list_tags],
    autodiff_backend='torch',   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
)
tnopt_psi.optimizer = 'adam'
TN_l = tnopt_psi.optimize(n=10)
tnopt_psi.optimizer = 'L-BFGS-B'
TN_l = tnopt_psi.optimize(n=10000)
#tnopt_psi.optimizer = 'adam'
#TN_l = tnopt_psi.optimize(n=2000)

print ( (E_exact-energy_f(TN_l, ham))/E_exact )


plt.loglog(tnopt_psi.losses, 'd', color = '#c30be3', label='D=2, lay=6')

plt.title('polyTN')
plt.ylabel('1-F')
plt.legend(loc='center left')
#plt.axhline(0.31078, color='black', label='D=2')
#plt.axhline(4.083e-05, color='black', label='D=8')
# plt.axhline(0.044, color='black', label='D=4')
# plt.axhline(0.001, color='black', label='D=8')

#, fontsize=60
#plt.xticks(size = 60)
#plt.yticks(size = 60)
#plt.ylim(0, 1)
#plt.xlim(0, 200, fontsize=60)
#plt.xlabel('Iteration',fontsize=60)
plt.grid(True)
plt.savefig('polyTN.pdf')
plt.clf()


