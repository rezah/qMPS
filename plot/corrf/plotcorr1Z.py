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


R=np.loadtxt("corrZf.txt")
x=R[:,2]
y=R[:,3]
y=[    abs(y[i])     for i in range(len(y))     ]


R=np.loadtxt("corrZQ4.txt")
xQ4=R[:,2]
yQ4=R[:,3]
errorQ4=[    abs((y[i]-yQ4[i])/ y[i])     for i in range(len(y))]
yQ4=[    abs(yQ4[i])     for i in range(len(yQ4))     ]






R=np.loadtxt("corrZQ5.txt")
xQ5=R[:,2]
yQ5=R[:,3]
errorQ5=[    abs((y[i]-yQ5[i])/ y[i])     for i in range(len(y))]
yQ5=[    abs(yQ5[i])     for i in range(len(yQ5))     ]



##print (errorD16)
R=np.loadtxt("corrZQ7.txt")
xQ7=R[:,2]
yQ7=R[:,3]
errorQ7=[    abs((y[i]-yQ7[i])/ y[i])     for i in range(len(y))]
yQ7=[    abs(yQ7[i])     for i in range(len(yQ7))     ]


##print (errorD16)
R=np.loadtxt("corrZQ8.txt")
xQ8=R[:,2]
yQ8=R[:,3]
errorQ8=[    abs((y[i]-yQ8[i])/ y[i])     for i in range(len(y))]
yQ8=[    abs(yQ8[i])     for i in range(len(yQ8))     ]


fig=plt.figure(figsize=(8,6))



#plt.plot( x, y, '--', color = '#e30b69',markersize=10, label=r'$D=16$')
plt.plot( x, yQ4, '>', color = '#a40000',markersize=11, label=r'$qMPS, q=4$')
plt.loglog( x, yQ5,'o',markersize=11, color = '#f57900', label=r'$qMPS, q=5$')
plt.loglog( x, yQ8,'s', color = '#e30b69',markersize=11, label=r'$qMPS, q=8$')
plt.loglog( x, y, '--', lw=2,color = '#204a87', label='DMRG')


plt.yscale("log")

#plt.title('qmps')
plt.ylabel(r'$C(r)$',fontsize=18)
plt.xlabel(r'$r$',fontsize=18)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

#plt.xlim([4,26])
#plt.ylim([ 1.e0, 4.e-2])


plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="lower left", prop={'size': 21})


plt.grid(True)
plt.savefig('corr1Z.pdf')
plt.clf()
