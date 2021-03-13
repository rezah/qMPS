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


R=np.loadtxt("q2lay4.txt")
xr10=R[:,0]
yr10=R[:,1]
errorr10=R[:,2]
errorr10=sort_low(errorr10)   

R=np.loadtxt("D4.txt")
xmera=R[:,0]
ymera=R[:,1]
errormera=R[:,2]
errormera=sort_low(errormera)   




R=np.loadtxt("q3lay8.txt")
xg10=R[:,0]
yg10=R[:,1]
errorg10=R[:,2]
errorg10=sort_low(errorg10)   


R=np.loadtxt("D8.txt")
xgmera=R[:,0]
ygmera=R[:,1]
errorgmera=R[:,2]
errorgmera=sort_low(errorgmera)   



#fig=plt.figure(figsize=(5,8))


plt.loglog( errorr10, '4', color = '#e3360b', label='n_q=2, lay=4')
#plt.loglog( errormera, '2', color = '#3465a4', label='D=4')

plt.loglog( errorg10, '>', color = '#0b8de3', label='n_q=3, lay=8')
#plt.loglog( errorgmera, 'o', color = '#72cfbb', label='D=8')


#plt.loglog( error11, '+', color = '#e3570b', label='lay=11')
#plt.loglog( error6, 'x', color = '#e30b69', label='lay=6')

#plt.loglog(  error4, '2', color = '#729fcf', label='lay=4')

#plt.loglog( error8, 'o', color = '#72cfbb', label='lay=8')

#plt.loglog( xq4l24, errorq4l24, '2', color = '#cf729d', label='q=4, lay=24')

plt.title('qmps')
plt.ylabel(r'$\delta$ E')
plt.xlabel(r'$n$')
plt.axhline(0.005450,color='black', label='mera, D=4')
plt.axhline(0.00070450, color='black', label='mera, D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('qmps-plot.pdf')
plt.clf()
