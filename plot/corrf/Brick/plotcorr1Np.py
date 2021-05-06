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


R=np.loadtxt("corrN2p.txt")
xl2p=R[:,2]
yl2p=R[:,3]
errorl2p=[    abs((y[i]-yl2p[i])/ y[i])     for i in range(len(y))]
yl2p=[    abs(yl2p[i])     for i in range(len(yl2p))     ]





R=np.loadtxt("corrN4.txt")
xl4=R[:,2]
yl4=R[:,3]
errorl4=[    abs((y[i]-yl4[i])/ y[i])     for i in range(len(y))]
yl4=[    abs(yl4[i])     for i in range(len(yl4))     ]



R=np.loadtxt("corrN4p.txt")
xl4p=R[:,2]
yl4p=R[:,3]
errorl4p=[    abs((y[i]-yl4p[i])/ y[i])     for i in range(len(y))]
yl4p=[    abs(yl4p[i])     for i in range(len(yl4p))     ]


R=np.loadtxt("corrN6p.txt")
xl6p=R[:,2]
yl6p=R[:,3]
errorl6p=[    abs((y[i]-yl6p[i])/ y[i])     for i in range(len(y))]
yl6p=[    abs(yl6p[i])     for i in range(len(yl6p))     ]



R=np.loadtxt("corrN8.txt")
xl8=R[:,2]
yl8=R[:,3]
errorl8=[    abs((y[i]-yl8[i])/ y[i])     for i in range(len(y))]
yl8=[    abs(yl8[i])     for i in range(len(yl8))     ]

R=np.loadtxt("corrN8p.txt")
xl8p=R[:,2]
yl8p=R[:,3]
errorl8p=[    abs((y[i]-yl8p[i])/ y[i])     for i in range(len(y))]
yl8p=[    abs(yl8p[i])     for i in range(len(yl8p))     ]



R=np.loadtxt("corrN10.txt")
xl10=R[:,2]
yl10=R[:,3]
errorl10=[    abs((y[i]-yl10[i])/ y[i])     for i in range(len(y))]
yl10=[    abs(yl10[i])     for i in range(len(yl10))     ]


R=np.loadtxt("corrN6.txt")
xl6=R[:,2]
yl6=R[:,3]
errorl6=[    abs((y[i]-yl6[i])/ y[i])     for i in range(len(y))]
yl6=[    abs(yl6[i])     for i in range(len(yl6))     ]




R=np.loadtxt("corrN12.txt")
xl12=R[:,2]
yl12=R[:,3]
errorl12=[    abs((y[i]-yl12[i])/ y[i])     for i in range(len(y))]
yl12=[    abs(yl12[i])     for i in range(len(yl12))     ]


R=np.loadtxt("corrNQ4.txt")
xQ4=R[:,2]
yQ4=R[:,3]
errorQ4=[    abs((y[i]-yQ4[i])/ y[i])     for i in range(len(y))]
yQ4=[    abs(yQ4[i])     for i in range(len(yQ4))     ]


R=np.loadtxt("corrNQ7.txt")
xQ7=R[:,2]
yQ7=R[:,3]
errorQ7=[    abs((y[i]-yQ7[i])/ y[i])     for i in range(len(y))]
yQ7=[    abs(yQ7[i])     for i in range(len(yQ7))     ]

fig=plt.figure(figsize=(8,6))



plt.loglog( x, y, '--', lw=2,color = '#204a87', label='DMRG')
#plt.plot( x, yQ4, '1', color = '#e30b69',markersize=22, label=r'$qMPS, q=4$')
#plt.plot( x, yQ7, 'o', color = '#e30b69',markersize=11, label=r'$qMPS, q=7$')

#plt.plot( x, yl4p, '-.',color = '#4e9a06',markersize=11, label=r'$QC-l, \tau=4$')
#plt.plot( x, yl6p, '-.D',color = '#f57900',markersize=11, label=r'$QC-l, \tau=6$')

plt.plot( x, yl4p, '-->', lw=3,color = '#e30b69',markersize=11, label=r'$ \tau=4$')

plt.plot( x, yl6p, '--h', lw=3,color = '#f57900',markersize=11, label=r'$ \tau=6$')

plt.plot( x, yl8p, '--o', lw=3,color = '#4e9a06',markersize=11, label=r'$ \tau=8$')
#plt.plot( x, yl10p, '--d', lw=3,color = '#cc0000',markersize=11, label=r'$ \tau=10$')

#plt.plot( x, yl12p, '-.s', color = '#a40000',markersize=11, label=r'$\tau=12$')


#plt.plot( x, yl8p, 'o', color = '#cc0000',markersize=10, label=r'$QC-l, \tau=8$')


plt.yscale("log")

#plt.title('qmps')
plt.ylabel(r'$C(r)$',fontsize=20)
plt.xlabel(r'$r$',fontsize=20)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

#plt.xlim([4,26])
plt.ylim([ 4.e-8, 1.e0])


plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="lower left", prop={'size': 21})


plt.grid(True)
plt.savefig('corr1Np.pdf')
plt.clf()
