from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib as mpl


mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1

def sort_low(error):
 val_min=1.e8
 error_f=[]
 for i in range(len(error)-1):
   if error[i]>=error[i+1] and  error[i]<=val_min :
    error_f.append(error[i])
    val_min=error[i+1]*1.0
    pass

 return error_f


R=np.loadtxt("q4lay14rand.txt")
xr10=R[:,0]
yr10=R[:,1]
errorr10=R[:,2]
errorr10=sort_low(errorr10)   


R=np.loadtxt("q4Lay6Random.txt")
xg10=R[:,0]
yg10=R[:,1]
errorg10=R[:,2]
errorg10=sort_low(errorg10)   

R=np.loadtxt("q4Lay10good.txt")
x11=R[:,0]
y11=R[:,1]
error11=R[:,2]
error11=sort_low(error11)   


R=np.loadtxt("q4Lay14good.txt")
x6=R[:,0]
y6=R[:,1]
error6=R[:,2]
error6=sort_low(error6)   



R=np.loadtxt("q4Lay18good.txt")
x4=R[:,0]
y4=R[:,1]
error4=R[:,2]
error4=sort_low(error4)   





plt.figure(figsize=(8, 6))
plt.axhline(0.00422,color='black', label='dMPS(DMRG), D=4')
plt.axhline(0.0001435, color='black', label='dMPS(DMRG), D=8')
plt.axhline(4.1178e-06, color='black', label='dMPS(DMRG), D=16')

plt.loglog( errorr10, '--', lw=4, color = '#e3360b', label=r'$ \tau=14,rand$')
plt.loglog( errorg10, '--', lw=4, color = '#72cfbb', label=r'$ \tau=6,rand$')
plt.loglog( error11, '--', lw=4, color = '#c74294', label=r'$ \tau=10,good$')
plt.loglog( error6, '--', lw=4, color = '#e3570b', label=r'$ \tau=14,good$')
plt.loglog(  error4, '--', lw=4, color = '#a40000', label=r'$ \tau=18,good$')
#plt.loglog( error8, 'o', color = '#72cfbb', label='lay=8')
#plt.loglog( xq4l24, errorq4l24, '2', color = '#cf729d', label='q=4, lay=24')




#plt.title('brickwall circuit')
plt.ylabel(r'$\delta$ E',fontsize=20)
plt.xlabel(r'$iterations$',fontsize=20)
#plt.axhline(0.00422,color='black', label='D=4')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="upper right", prop={'size': 16})


plt.xlim([1,4.e5])
plt.ylim([1.0e-6, 30.e-1])

#plt.grid(True)
plt.savefig('qmps-plot.pdf')
plt.clf()
