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


R=np.loadtxt("corrNf.txt")
x=R[:,2]
y=R[:,3]
y=[    abs(y[i])     for i in range(len(y))     ]


R=np.loadtxt("corrN2.txt")
xl2=R[:,2]
yl2=R[:,3]
errorl2=[    abs((y[i]-yl2[i])/ y[i])     for i in range(len(y))]
yl2=[    abs(yl2[i])     for i in range(len(yl2))     ]





R=np.loadtxt("corrN4.txt")
xl4=R[:,2]
yl4=R[:,3]
errorl4=[    abs((y[i]-yl4[i])/ y[i])     for i in range(len(y))]
yl4=[    abs(yl4[i])     for i in range(len(yl4))     ]




R=np.loadtxt("corrN6.txt")
xl6=R[:,2]
yl6=R[:,3]
errorl6=[    abs((y[i]-yl6[i])/ y[i])     for i in range(len(y))]
yl6=[    abs(yl6[i])     for i in range(len(yl6))     ]


R=np.loadtxt("corrN8.txt")
xl8=R[:,2]
yl8=R[:,3]
errorl8=[    abs((y[i]-yl8[i])/ y[i])     for i in range(len(y))]
yl8=[    abs(yl8[i])     for i in range(len(yl8))     ]



R=np.loadtxt("corrN10.txt")
xl10=R[:,2]
yl10=R[:,3]
errorl10=[    abs((y[i]-yl10[i])/ y[i])     for i in range(len(y))]
yl10=[    abs(yl10[i])     for i in range(len(yl10))     ]






R=np.loadtxt("corrN12.txt")
xl12=R[:,2]
yl12=R[:,3]
errorl12=[    abs((y[i]-yl12[i])/ y[i])     for i in range(len(y))]
yl12=[    abs(yl12[i])     for i in range(len(yl12))     ]




fig=plt.figure(figsize=(8,6))



plt.loglog( x, y, '--', lw=3,color = '#204a87', label='MPS-DMRG')
plt.plot( x, yl6, '1', color = '#e30b69',markersize=10, label=r'$qMPS, \tau=6$')
plt.plot( x, yl8, 'o', color = '#e30b69',markersize=10, label=r'$qMPS, \tau=8$')
plt.plot( x, yl10, '>', color = '#e30b69',markersize=10, label=r'$qMPS, \tau=10$')
plt.plot( x, yl12, 's', color = '#204a87',markersize=10, label=r'$qMPS, \tau=12$')


plt.yscale("log")

#plt.title('qmps')
plt.ylabel(r'$C(r)$',fontsize=16)
plt.xlabel(r'$r$',fontsize=16)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

#plt.xlim([4,26])
plt.ylim([ 1.e0, 4.e-10])


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc="lower left", prop={'size': 20})


plt.grid(True)
plt.savefig('corr1N.pdf')
plt.clf()
