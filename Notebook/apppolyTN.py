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
d=2
D=2
chi=60
L_x=24
L_y=5
seed_0=4
#data_type='complex128'   #'float64'
data_type='float64'   #'float64'

class    polymps:

 def __init__(self, d=2, L_y=4, L_x=10, dist_type='uniform', data_type='float64', seed_0=10):
  self.L_x = L_x
  self.d = d
  self.L_y = L_y
  self.type=data_type
  self.dist=dist_type
  
  
  peps = qtn.PEPS.rand(Lx=L_x, Ly=L_y, bond_dim=D, phys_dim=d, seed=4)

  rotate_ten_list=[]
  for j in range(L_y-1):
   for i in range(L_x):
    A=qtn.Tensor(qu.rand(2, seed=seed_0+i+j, dist='uniform', dtype=data_type).reshape(2), inds={f"k{i},{j}"}, tags={})
    rotate_ten_list.append(A)

  R_l=qtn.TensorNetwork(rotate_ten_list)
  self.tn=peps & R_l
  for j in range(L_y-1):
   for i in range(L_x):
    self.tn.contract_ind(f"k{i},{j}", optimize='auto-hq')

  #TN_l.graph(color=peps.site_tags, show_tags=True, figsize=(10, 10))

  for j in range(L_y):
   for i in range(L_x):
     t=list((self.tn[{ f'I{i},{j}', f'ROW{i}', f'COL{j}'}].inds))
     for m in range(len(t)):
      if t[m]==f'k{i},{j}':
          t[m]=f'k{i}'
     self.tn[{ f'I{i},{j}', f'ROW{i}', f'COL{j}'}].modify(inds=t)            



#   for j in range(L_y):
#    for i in range(L_x):
#      dim_tuple=self.tn[{ f'I{i},{j}', f'ROW{i}', f'COL{j}' }].shape
#      lis_a=list(dim_tuple)
#      dim=1
#      for i_val in range(len(lis_a)):
#       dim*=lis_a[i_val]
#      rand_tn=qu.rand(dim, dist='uniform', seed=seed_0, dtype=data_type).reshape(*dim_tuple)
#      rand_tn=rand_tn*LA.norm(rand_tn)**(-1.0) 
#      self.tn[{f'COL{j}', f'I{i},{j}', f'ROW{i}' }].modify(data=rand_tn)

  self.tn.balance_bonds_()
  self.tn.equalize_norms_(2.0)

 def norm(self, chi):
    TN_l_h=self.tn.H
    for j in range(self.L_y):
     for i in range(self.L_x):
       TN_l_h[{ f'I{i},{j}', f'ROW{i}', f'COL{j}' }].modify(tags={ f'I{i},{-j+2*self.L_y-1}', f'ROW{i}', f'COL{-j+2*self.L_y-1}'})

    Norm_val_tn=TN_l_h & self.tn
    Norm_peps=Norm_val_tn.view_as(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=self.L_x,
        Ly=2*self.L_y,
        site_tag_id='I{},{}',
        row_tag_id='ROW{}',
        col_tag_id='COL{}',
    )

    return Norm_peps.contract_boundary(sequence='l', max_bond=chi, cutoff=0.0, method='svd')

 def get_tags_polyTN(self):
  list_tag=[]
  for j in range(L_y):
   for i in range(L_x):
     list_tag.append({ f'I{i},{j}', f'ROW{i}', f'COL{j}' })
  return list_tag

 def normalize(self, chi):
    TN_l_h=self.tn.H
    for j in range(self.L_y):
     for i in range(self.L_x):
       TN_l_h[{ f'I{i},{j}', f'ROW{i}', f'COL{j}' }].modify(tags={ f'I{i},{-j+2*self.L_y-1}', f'ROW{i}', f'COL{-j+2*self.L_y-1}'})

    Norm_val_tn=TN_l_h & self.tn
    Norm_peps=Norm_val_tn.view_as(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=self.L_x,
        Ly=2*self.L_y,
        site_tag_id='I{},{}',
        row_tag_id='ROW{}',
        col_tag_id='COL{}',
    )
    Norm_val=Norm_peps.contract_boundary(sequence='t', max_bond=chi, cutoff=0.0, method='svd')
    print ("norm, t", Norm_val)
    Norm_val_l=Norm_peps.contract_boundary(sequence='l', max_bond=chi, cutoff=0.0, method='svd')
    print ("norm, l", Norm_val_l)
    Norm_val_r=Norm_peps.contract_boundary(sequence='r', max_bond=chi, cutoff=0.0, method='svd')
    print ("norm,r",  Norm_val_r)
    Norm_val_b=Norm_peps.contract_boundary(sequence='b', max_bond=chi, cutoff=0.0, method='svd')
    print ("norm, b", Norm_val_b)
    self.tn=self.tn*(Norm_val**(-0.5))




 def __setitem__(self, pos, tn_ext):
  self.tn[pos].modify(data=tn_ext.data)



 def __getitem__(self, pos):
  return  self.tn[pos]



 def draw(self):
  return self.tn.graph(color=[f'ROW{i}' for i in range(self.L_x)], show_tags=False, figsize=(10, 10))

 def energy(self, ham, chi):

        
    val_norm=self.norm(chi) 
    self.tn=self.tn*(val_norm**(-.5))
    TN_l_ham=self.tn.H


    for j in range(self.L_y):
     for i in range(self.L_x):
       TN_l_ham[{ f'I{i},{j}', f'ROW{i}', f'COL{j}' }].modify(tags={ f'I{i},{-j+2*self.L_y}', f'ROW{i}', f'COL{-j+2*self.L_y}'})


    for j in range( self.L_y):
     for i in range(self.L_x):
       t=list((TN_l_ham[{ f'I{i},{-j+2*self.L_y}', f'ROW{i}', f'COL{-j+2*self.L_y}'}].inds))
       for m in range(len(t)):
        if t[m]==f'k{i}':
              t[m]=f'b{i}'
       TN_l_ham[{ f'I{i},{-j+2*self.L_y}', f'ROW{i}', f'COL{-j+2*self.L_y}'}].modify(inds=t)            
#    ham_f=ham*1.0
#    for i in range(self.L_x):
#     ham_f[i].modify( tags=[ f'I{i},{self.L_y}', f'ROW{i}' , f'COL{self.L_y}', 'ham' ]  )


    Ham_val_tn=(TN_l_ham | ham | self.tn)
    Ham_peps=Ham_val_tn.view_as(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=self.L_x,
        Ly=2*self.L_y+1,
        site_tag_id='I{},{}',
        row_tag_id='ROW{}',
        col_tag_id='COL{}',
    )



    return  Ham_peps.contract_boundary(sequence='t', max_bond=chi, cutoff=0.0, method='svd')




    


 def auto_diff_energy(self, ham, chi):
  TN_l=self.tn


  def norm_TN(TN_l,chi):

    TN_l_h=TN_l.H
    for j in range(L_y):
     for i in range(L_x):
       TN_l_h[{ f'I{i},{j}', f'ROW{i}', f'COL{j}' }].modify(tags={ f'I{i},{-j+2*L_y-1}', f'ROW{i}', f'COL{-j+2*L_y-1}'})
    Norm_val_tn=TN_l_h & TN_l
    Norm_peps=Norm_val_tn.view_as(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=L_x,
        Ly=2*L_y,
        site_tag_id='I{},{}',
        row_tag_id='ROW{}',
        col_tag_id='COL{}')

    norm_val=Norm_peps.contract_boundary(sequence='l', max_bond=chi, cutoff=0.0, method='svd')
    #print ("norm_val", norm_val)
    TN_l=TN_l*(norm_val**(-0.5))
    return TN_l




  def energy_f(TN_l, ham, L_x, L_y, chi):
    #print (chi)
    #print (TN_l.exponent())
    TN_l.balance_bonds_()
    #TN_l.equalize_norms_(1.)
    TN_l=norm_TN(TN_l,chi)
    TN_l_ham=TN_l.H



    for j in range(L_y):
     for i in range(L_x):
       TN_l_ham[{ f'I{i},{j}', f'ROW{i}', f'COL{j}' }].modify(tags={ f'I{i},{-j+2*L_y}', f'ROW{i}', f'COL{-j+2*L_y}'})


    for j in range(L_y):
     for i in range(L_x):
       t=list((TN_l_ham[{ f'I{i},{-j+2*L_y}', f'ROW{i}', f'COL{-j+2*L_y}'}].inds))
       for m in range(len(t)):
        if t[m]==f'k{i}':
              t[m]=f'b{i}'
       TN_l_ham[{ f'I{i},{-j+2*L_y}', f'ROW{i}', f'COL{-j+2*L_y}'}].modify(inds=t)            

    Ham_val_tn=TN_l_ham & ham & TN_l
    Ham_peps=Ham_val_tn.view_as(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=L_x,
        Ly=2*L_y+1,
        site_tag_id='I{},{}',
        row_tag_id='ROW{}',
        col_tag_id='COL{}',
    )

    return  Ham_peps.contract_boundary(sequence='l', max_bond=chi, cutoff=0.0, method='svd')



    
    
    
  TN_l=self.tn
  
  print (energy_f(TN_l, ham, L_x, L_y,chi))
  tnopt = qtn.TNOptimizer(
        TN_l,                                      # the initial TN
        loss_fn=energy_f,                         # the loss function
        constant_tags=[],                   # only optimize the tensors tagged 'G'
        loss_constants={'ham': ham},  # additional tensor/tn kwargs
        loss_kwargs={'L_x':self.L_x, 'L_y':self.L_y, "chi":chi },
        tags=[],
        optimizer='L-BFGS-B',             # how to optimize
        autodiff_backend='torch',       # how to generate/compute the gradient     
     )
#jit_fn=True,
  #print ('adam')
  #tnopt.optimizer = 'adam' 
  TN_l = tnopt.optimize(n=10)

#  print ('L-BFGS-B')
#  tnopt.optimizer = 'L-BFGS-B'
#  TN_l = tnopt.optimize(n=10)
#  TN_l = tnopt.optimize(1000, tol=1e-12, ftol=1e-12, gtol=1e-12)

  list_tag=TN_l_class.get_tags_polyTN()
  for i in list_tag:
    TN_l_class[i]=TN_l[i]

  save_to_disk(TN_l_class, "Data/TN_l_class")
  save_to_disk(TN_l, "Data/TN_l")

  plt.plot(tnopt.losses, 'd', color = '#c30be3', label='L=20, lay=4')

  plt.title('polyenergy')
  plt.ylabel('1-f')
  plt.legend(loc='center left')

  plt.grid(True)
  plt.savefig('polyenergy.pdf')
  plt.clf()



 def auto_diff_fidel(self, psi, chi):
  TN_l=self.tn


  def norm_TN(TN_l,chi):

    TN_l_h=TN_l.H
    for j in range(L_y):
     for i in range(L_x):
       TN_l_h[{ f'I{i},{j}', f'ROW{i}', f'COL{j}' }].modify(tags={ f'I{i},{-j+2*L_y-1}', f'ROW{i}', f'COL{-j+2*L_y-1}'})
    Norm_val_tn=TN_l_h & TN_l
    Norm_peps=Norm_val_tn.view_as(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=L_x,
        Ly=2*L_y,
        site_tag_id='I{},{}',
        row_tag_id='ROW{}',
        col_tag_id='COL{}')

    norm_val=Norm_peps.contract_boundary(sequence='r', max_bond=chi, cutoff=0.0, method='svd')
    #print ("norm_val", norm_val)
    TN_l=TN_l*(norm_val**(-0.5))
    return TN_l




  def fidel_f(TN_l, psi, L_x, L_y, chi):
    #print (chi)
    #print (TN_l.exponent())
    TN_l.balance_bonds_()
    #TN_l.equalize_norms_()
    TN_l=norm_TN(TN_l,chi)
    TN_l_ham=TN_l.H

       

    Ham_val_tn=psi.H & TN_l
    Ham_peps=Ham_val_tn.view_as(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=L_x,
        Ly=L_y+1,
        site_tag_id='I{},{}',
        row_tag_id='ROW{}',
        col_tag_id='COL{}',
    )

    return  1-abs(Ham_peps.contract_boundary(sequence='r', max_bond=chi, cutoff=0.0, method='svd'))


  TN_l=self.tn
  
  print (fidel_f(TN_l, psi, L_x, L_y,chi))
  tnopt = qtn.TNOptimizer(
        TN_l,                                      # the initial TN
        loss_fn=fidel_f,                         # the loss function
        constant_tags=[],                   # only optimize the tensors tagged 'G'
        loss_constants={'psi': psi},  # additional tensor/tn kwargs
        loss_kwargs={'L_x':self.L_x, 'L_y':self.L_y, "chi":chi },
        tags=[],
        optimizer='L-BFGS-B',             # how to optimize
        autodiff_backend='torch',       # how to generate/compute the gradient     
     )
#jit_fn=True,
  #print ('adam')
  #print ('lbgs_b')
  #tnopt.optimizer = 'adam' 
  TN_l = tnopt.optimize(n=800)

#  print ('L-BFGS-B')
#  tnopt.optimizer = 'L-BFGS-B'
#  TN_l = tnopt.optimize(n=10)
#  TN_l = tnopt.optimize(1000, tol=1e-12, ftol=1e-12, gtol=1e-12)

  list_tag=TN_l_class.get_tags_polyTN()
  for i in list_tag:
    TN_l_class[i]=TN_l[i]

  save_to_disk(TN_l_class, "Data/TN_l_class")
  save_to_disk(TN_l, "Data/TN_l")
  y=tnopt.losses[:]
  file = open("Data/y"+ str(L_y)+".txt", "w")
  for index in range(len(y)):
     file.write(str(index) + "  "+ str(y[index]) + "\n")
  file.close()

  
  plt.loglog(y, 'd', color = '#c30be3', label='L=32, lay=2')

  plt.title('polyfidel')
  plt.ylabel('1-F')
  plt.legend(loc='center left')

  plt.grid(True)
  plt.savefig('polyfidel.pdf')
  plt.clf()










ham = MPO_ham_heis(L_x, j=1.0, bz=0.0, S=0.5, cyclic=False)
dmrg = DMRG2(ham, bond_dims=[10, 26, 60, 80, 100, 200, 200, 300], cutoffs=1e-10)
dmrg.solve(tol=1e-8, verbosity=0 )
print("DMRG=", dmrg.energy)
E_exact=dmrg.energy
psi=dmrg.state
print("psi=", psi.show())

E_exact
ham=ham*1.0
for i in range(L_x):
 ham[i].modify( tags=[ f'I{i},{L_y}', f'ROW{i}' , f'COL{L_y}', 'ham' ]  )
for i in range(L_x):
 psi[i].modify( tags=[ f'I{i},{L_y}', f'ROW{i}' , f'COL{L_y}', 'ham' ]  )


#print (get_contract_backend())


#set_contract_backend("numpy")
#set_tensor_linop_backend("numpy")

#print (get_contract_backend())


#set_contract_backend("slepc")
#set_tensor_linop_backend("slepc")






TN_l_class=polymps(d=2, L_x=L_x, L_y=L_y, seed_0=2, data_type='float64')


TN_l_class=load_from_disk("Data/TN_l_class")



TN_l_class.normalize(chi=chi)
print ("L_x, L_y, chi, D", L_x, L_y, chi, D)
print (TN_l_class.norm(chi), TN_l_class.energy(ham, chi))


TN_l_class.auto_diff_fidel(psi, chi)

TN_l_class.auto_diff_energy(ham, chi)



