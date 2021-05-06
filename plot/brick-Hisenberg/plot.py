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


R=np.loadtxt("lay10rand.txt")
xr10=R[:,0]
yr10=R[:,1]
errorr10=R[:,2]
errorr10=sort_low(errorr10)   


R=np.loadtxt("lay10good.txt")
xg10=R[:,0]
yg10=R[:,1]
errorg10=R[:,2]
errorg10=sort_low(errorg10)   

R=np.loadtxt("lay11good.txt")
x11=R[:,0]
y11=R[:,1]
error11=R[:,2]
error11=sort_low(error11)   


R=np.loadtxt("lay6good.txt")
x6=R[:,0]
y6=R[:,1]
error6=R[:,2]
error6=sort_low(error6)   



R=np.loadtxt("lay4good.txt")
x4=R[:,0]
y4=R[:,1]
error4=R[:,2]
error4=sort_low(error4)   



R=np.loadtxt("lay8good.txt")
x8=R[:,0]
y8=R[:,1]
error8=R[:,2]
error8=sort_low(error8)   

R=np.loadtxt("lay9good.txt")
x9=R[:,0]
y9=R[:,1]
error9=R[:,2]
error9=sort_low(error9)   



R=np.loadtxt("lay8rand.txt")
x8r=R[:,0]
y8r=R[:,1]
error8r=R[:,2]
error8r=sort_low(error8r)   


R=np.loadtxt("lay12good.txt")
x12=R[:,0]
y12=R[:,1]
error12=R[:,2]
error12=sort_low(error12)   


R=np.loadtxt("lay13good.txt")
x13=R[:,0]
y13=R[:,1]
error13=R[:,2]
error13=sort_low(error13)   


plt.figure(figsize=(8, 6))


#plt.loglog( errorr10, '4', color = '#e3360b', label='lay=10, random')
plt.loglog( error8, '-', lw=4,color = '#72cfbb', label=r'$\tau=8$')
#plt.loglog( error9, '3', color = '#cf729d', label='lay=9, good')
plt.loglog( errorg10, '--',lw=4, color = '#0b8de3', label=r'$\tau=10$')
plt.loglog( error11, '-.', lw=4,color = '#e3570b', label=r'$\tau=11$')
plt.loglog( error12, ':',lw=4, color = '#729fcf', label=r'$\tau=12$')
plt.loglog( error13, '--', lw=4,color = '#96068a', label=r'$\tau=13$')

#plt.loglog( error6, 'x', color = '#e30b69', label='lay=6')

#plt.loglog(  error4, '2', color = '#729fcf', label='lay=4')


#plt.loglog( error8r, '2', color = '#cf729d', label='lay=8, random')

#plt.title('brickwall circuit')
plt.ylabel(r'$\delta$ E',fontsize=20)
plt.xlabel(r'$iterations$',fontsize=20)
#plt.axhline(0.00422,color='black', label='D=4')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="upper right", prop={'size': 18})

plt.xlim([1,5.5e3])
plt.ylim([4.2e-3, 3.e-2])


#plt.grid(True)
plt.savefig('qmps-plot.pdf')
plt.clf()
