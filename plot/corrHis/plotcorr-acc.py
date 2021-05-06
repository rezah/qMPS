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


R=np.loadtxt("corrZDMRG.txt")
x=R[:,2]
y=R[:,3]




R=np.loadtxt("corrZl4p.txt")
xl4p=R[:,2]
yl4p=R[:,3]
errorl4p=[    abs((y[i]-yl4p[i])/ y[i])     for i in range(len(yl4p))]
yl4p=[    abs(yl4p[i])     for i in range(len(yl4p))     ]


R=np.loadtxt("corrZl6p.txt")
xl6p=R[:,2]
yl6p=R[:,3]
errorl6p=[    abs((y[i]-yl6p[i])/ y[i])     for i in range(len(yl6p))]
yl6p=[    abs(yl6p[i])     for i in range(len(yl6p))     ]


R=np.loadtxt("corrZl7p.txt")
xl7p=R[:,2]
yl7p=R[:,3]
errorl7p=[    abs((y[i]-yl7p[i])/ y[i])     for i in range(len(yl7p))]
yl7p=[    abs(yl7p[i])     for i in range(len(yl7p))     ]



R=np.loadtxt("corrZl8.txt")
xl8=R[:,2]
yl8=R[:,3]
errorl8=[    abs((y[i]-yl8[i])/ y[i])     for i in range(len(yl8))]
yl8=[    abs(yl8[i])     for i in range(len(yl8))     ]


R=np.loadtxt("corrZl12.txt")
xl12=R[:,2]
yl12=R[:,3]
errorl12=[    abs((y[i]-yl12[i])/ y[i])     for i in range(len(yl12))]
yl12=[    abs(yl12[i])     for i in range(len(yl12))     ]




R=np.loadtxt("corrZDMRGD16.txt")
xD16=R[:,2]
yD16=R[:,3]
errorD16=[    abs((y[i]-yD16[i])/ y[i])     for i in range(len(y))]

print (errorD16)

R=np.loadtxt("corrZDMRGD32.txt")
xD32=R[:,2]
yD32=R[:,3]
errorD32=[    abs((y[i]-yD32[i])/ y[i])     for i in range(len(y))]



R=np.loadtxt("corrZPQMPSq5l12.txt")
xq5p=R[:,2]
yq5p=R[:,3]
errorq5p=[    abs((y[i]-yq5p[i])/ y[i])     for i in range(len(y))]



R=np.loadtxt("corrZqMPSq5lay14.txt")
xq5b=R[:,2]
yq5b=R[:,3]
errorq5b=[    abs((y[i]-yq5b[i])/ y[i])     for i in range(len(y))]

R=np.loadtxt("corrZQMPSq4l20.txt")
xq4b=R[:,2]
yq4b=R[:,3]
errorq4b=[    abs((y[i]-yq4b[i])/ y[i])     for i in range(len(y))]





plt.figure(figsize=(8, 6))



plt.plot( x, errorD16, '-o',  markersize=12, color = '#729fcf', label=r'$D=16$')

plt.plot( x, errorq4b, '--h', color = '#e30b69',markersize=12, label=r'$qMPS, n_q=4$')
#plt.loglog( x, errorq5p,'3', color = '#cf729d',markersize=15, label=r'$qMPS, n_q=5$')
#plt.loglog( x, errorq5b,'-D',markersize=12, color = '#c74294', label=r'$qMPS, n_q=5$')
plt.loglog( x, errorD32, '-p', markersize=12,color = '#204a87', label='MPS, D=32')








plt.yscale("log")

#plt.title('qmps')
plt.ylabel(r'$\delta$ C',fontsize=20)
plt.xlabel(r'$r$',fontsize=20)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.xlim([1,26])
plt.ylim([ 1.e-1, 5.e-6])


plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(loc="lower left", prop={'size': 20})


plt.grid(True)
plt.savefig('qmpsCorracc.pdf')
plt.clf()
