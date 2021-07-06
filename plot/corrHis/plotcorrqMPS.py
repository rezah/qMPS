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



R=np.loadtxt("corrZDMRGD16.txt")
xD16=R[:,2]
yD16=R[:,3]
errorD16=[    abs((abs(y[i])-abs(yD16[i]))/ y[i])     for i in range(len(y))]



R=np.loadtxt("corrZDMRGD32.txt")
xD32=R[:,2]
yD32=R[:,3]
errorD32=[    abs((abs(y[i])-abs(yD32[i]))/ y[i])     for i in range(len(y))]
yD32=[    abs(yD32[i])     for i in range(len(yD32))     ]






R=np.loadtxt("corrZQMPSq4l20.txt")
xq4b=R[:,2]
yq4b=R[:,3]
errorq4b=[    abs((abs(y[i])-abs(yq4b[i]))/ y[i])     for i in range(len(y))]
yq4b=[    abs(yq4b[i])     for i in range(len(yq4b))     ]



R=np.loadtxt("corrZq4l4.txt")
xq4l4b=R[:,2]
yq4l4b=R[:,3]
errorq4l4b=[    abs((abs(y[i])-abs(yq4l4b[i]))/ y[i])     for i in range(len(y))]
yq4l4b=[    abs(yq4l4b[i])     for i in range(len(yq4l4b))     ]


R=np.loadtxt("corrZq4l5.txt")
xq4l5b=R[:,2]
yq4l5b=R[:,3]
errorq4l5b=[    abs((abs(y[i])-abs(yq4l5b[i]))/ y[i])     for i in range(len(y))]
yq4l5b=[    abs(yq4l5b[i])     for i in range(len(yq4l5b))     ]



R=np.loadtxt("corrZq4l8.txt")
xq4l8b=R[:,2]
yq4l8b=R[:,3]
errorq4l8b=[    abs((abs(y[i])-abs(yq4l8b[i]))/ y[i])     for i in range(len(y))]
yq4l8b=[    abs(yq4l8b[i])     for i in range(len(yq4l8b))     ]




R=np.loadtxt("corrZq4l12.txt")
xq4l12b=R[:,2]
yq4l12b=R[:,3]
errorq4l12b=[    abs((abs(y[i])-abs(yq4l12b[i]))/ y[i])     for i in range(len(y))]
yq4l12b=[    abs(yq4l12b[i])     for i in range(len(yq4l12b))     ]



R=np.loadtxt("corrZq8l4.txt")
xq8l4b=R[:,2]
yq8l4b=R[:,3]
errorq8l4b=[    abs((abs(y[i])-abs(yq8l4b[i]))/ y[i])     for i in range(len(y))]
yq8l4b=[    abs(yq8l4b[i])     for i in range(len(yq8l4b))     ]


R=np.loadtxt("corrZPQMPSq5l12.txt")
xq5p=R[:,2]
yq5p=list(R[:,3])
errorq5p=[    abs((abs(y[i])-abs(yq5p[i]))/ y[i])     for i in range(len(y))]
yq5p=[    abs(yq5p[i])     for i in range(len(yq5p))     ]


R=np.loadtxt("corrZqMPSq5lay14.txt")
xq5b=R[:,2]
yq5b=R[:,3]
errorq5b=[    abs((abs(y[i])-abs(yq5b[i]))/ y[i])     for i in range(len(y))]
yq5b=[    abs(yq5b[i])     for i in range(len(yq5b))     ]


R=np.loadtxt("corrZq9l6.txt")
xq9l6b=R[:,2]
yq9l6b=R[:,3]
yq9l6b=[    abs(yq9l6b[i])     for i in range(len(yq9l6b))     ]
errorq9l6b=[    abs((abs(y[i])-abs(yq9l6b[i]))/ y[i])     for i in range(len(y))]


plt.figure(figsize=(8, 6))


#plt.loglog( x, y, '--', lw=2,color = '#204a87', label='DMRG')
#plt.plot( x, errorD16, '--', color = '#204a87',markersize=10, label=r'dMPS(DMRG), $D=16$')
#plt.plot( x, yq4b, '1', color = '#e30b69',markersize=22, label=r'qMPS, $q=4$')


#plt.plot( xq4l4b, yq4l4b, '-.P', color = '#cf729d',markersize=11, label=r'qMPS, $q=4, \tau=4$')
#plt.plot( xq4l8b, yq4l8b, '-.h', color = '#e30b69',markersize=11, label=r'qMPS, $q=4, \tau=8$')
#plt.plot( xq4l12b, yq4l12b, '-.o', color = '#e9b96e',markersize=11, label=r'qMPS, $q=4, \tau=12$')
#plt.plot( xq8l4b, yq8l4b, '-.s', color = '#75507b',markersize=11, label=r'qMPS, $q=8, \tau=4$')


plt.plot( xq4l4b, errorq4l4b, '-->', color = '#cf729d',markersize=11, label=r'qMPS, $q=4, \tau=4$')
#plt.plot( xq4l5b, errorq4l5b, '-.s', color = '#72cfbb',markersize=11, label=r'qMPS, $q=4, \tau=5$')
plt.plot( xq4l8b, errorq4l8b, '-->', color = '#e9b96e',markersize=11, label=r'qMPS, $q=4, \tau=8$')
#plt.plot( xq4l12b, errorq4l12b, '-->', color = '#ce5c00',markersize=11, label=r'qMPS, $q=4, \tau=12$')
#plt.plot( xq4b, errorq4b, '-->', color = '#a40000',markersize=11, label=r'qMPS, $q=4, \tau=18$')
#plt.plot( xq5p, errorq5p, '-.>', color = '#e30b69',markersize=11, label=r'qMPS, $q=5, \tau=12$')
#plt.plot( xq5b, errorq5b, '-.<', color = '#f12626',markersize=11, label=r'qMPS, $q=5, \tau=14$')
plt.plot( xq8l4b, errorq8l4b, '-.s', color = '#ce5c00',markersize=11, label=r'qMPS, $q=8, \tau=4$')
plt.plot( xq9l6b, errorq9l6b, '-.o', color = '#30bde4',markersize=11, label=r'qMPS, $q=9, \tau=6$')


#plt.loglog( x, yq5p,'3', color = '#cf729d',markersize=15, label=r'$qMPS, n_q=5$')
#plt.loglog( x, yq5b,'2',markersize=15, color = '#c74294', label=r'$qMPS, n_q=5$')


plt.yscale("log")

#plt.title('qmps')
plt.ylabel(r'$\delta C(r)$',fontsize=20)
plt.xlabel(r'$r$',fontsize=20)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.xlim([0.2,23])
plt.ylim([ 4.e-6, 0.8e0])


plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(loc="lower right", prop={'size': 16})


plt.grid(True)
plt.savefig('corrqMPS.pdf')
plt.clf()
