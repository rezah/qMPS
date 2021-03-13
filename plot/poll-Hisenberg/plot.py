from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA

def sort_low(error):
 val_min=1.e8
 error_f=[]
 for i in range(len(error)-1):
   if error[i]>=error[i+1] and  error[i]<=val_min :
    error_f.append(error[i])
    val_min=error[i+1]*1.0
    pass

 return error_f



R=np.loadtxt("l5good.txt")
x5=R[:,0]
y5=R[:,1]
error5=R[:,2]
error5=sort_low(error5)   


R=np.loadtxt("l6good.txt")
x6=R[:,0]
y6=R[:,1]
error6=R[:,2]
error6=sort_low(error6)   



R=np.loadtxt("l7good.txt")
x7=R[:,0]
y7=R[:,1]
error7=R[:,2]
error7=sort_low(error7)   



R=np.loadtxt("l8good.txt")
x8=R[:,0]
y8=R[:,1]
error8=R[:,2]
error8=sort_low(error8)   

R=np.loadtxt("l8rand.txt")
x8r=R[:,0]
y8r=R[:,1]
error8r=R[:,2]
error8r=sort_low(error8r)   




#fig=plt.figure(figsize=(5,8))


#plt.loglog( errorr10, '4', color = '#e3360b', label='lay=10, random')
#plt.loglog( errorg10, '>', color = '#0b8de3', label='lay=10, good')

#plt.loglog( error11, '+', color = '#e3570b', label='lay=11')

plt.loglog( error6, 'x', color = '#e30b69', label='lay=6')

plt.loglog(  error8r, '2', color = '#729fcf', label='lay=8, random')

plt.loglog( error8, 'o', color = '#72cfbb', label='lay=8')

plt.loglog( error5, '2', color = '#cf729d', label=' lay=5')

#plt.title('qmps')
plt.ylabel(r'$\delta$ E')
plt.xlabel(r'$n$')
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('qmps-plot.pdf')
plt.clf()
