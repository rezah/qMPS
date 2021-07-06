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
y=[    abs(y[i])     for i in range(len(y))     ]



R=np.loadtxt("corrZq2M.txt")
xq2M=R[:,2]
yq2M=R[:,3]
yq2M=[    abs(yq2M[i])     for i in range(len(yq2M))     ]
errorq2M=[    abs((abs(y[i])-abs(yq2M[i]))/ y[i])     for i in range(len(yq2M))]



R=np.loadtxt("corrZq3M.txt")
xq3M=R[:,2]
yq3M=R[:,3]
yq3M=[    abs(yq3M[i])     for i in range(len(yq3M))     ]
errorq3M=[    abs((abs(y[i])-abs(yq3M[i]))/ y[i])     for i in range(len(yq3M))]



R=np.loadtxt("corrZl4p.txt")
xl4p=R[:,2]
yl4p=R[:,3]
yl4p=[    abs(yl4p[i])     for i in range(len(yl4p))     ]
errorl4p=[    abs((y[i]-yl4p[i])/ y[i])     for i in range(len(yl4p))]


R=np.loadtxt("corrZl6p.txt")
xl6p=R[:,2]
yl6p=R[:,3]
yl6p=[    abs(yl6p[i])     for i in range(len(yl6p))     ]
errorl6p=[    abs((y[i]-yl6p[i])/ y[i])     for i in range(len(yl6p))]


R=np.loadtxt("corrZl7p.txt")
xl7p=R[:,2]
yl7p=R[:,3]
yl7p=[    abs(yl7p[i])     for i in range(len(yl7p))     ]
errorl7p=[    abs((y[i]-yl7p[i])/ y[i])     for i in range(len(yl7p))]



R=np.loadtxt("corrZl8.txt")
xl8=R[:,2]
yl8=R[:,3]
yl8=[    abs(yl8[i])     for i in range(len(yl8))     ]
errorl8=[    abs((y[i]-yl8[i])/ y[i])     for i in range(len(yl8))]


R=np.loadtxt("corrZl12.txt")
xl12=R[:,2]
yl12=R[:,3]
yl12=[    abs(yl12[i])     for i in range(len(yl12))     ]
errorl12=[    abs((y[i]-yl12[i])/ y[i])     for i in range(len(yl12))]



R=np.loadtxt("corrZDMRGD16.txt")
xD16=R[:,2]
yD16=R[:,3]
yD16=[    abs(yD16[i])     for i in range(len(yD16))     ]
errorD16=[    abs((y[i]-yD16[i])/ y[i])     for i in range(len(y))]

#print (errorD16)

R=np.loadtxt("corrZDMRGD32.txt")
xD32=R[:,2]
yD32=R[:,3]
yD32=[    abs(yD32[i])     for i in range(len(yD32))     ]
errorD32=[    abs((y[i]-yD32[i])/ y[i])     for i in range(len(y))]



R=np.loadtxt("corrZPQMPSq5l12.txt")
xq5p=R[:,2]
yq5p=list(R[:,3])
yq5p=[    abs(yq5p[i])     for i in range(len(yq5p))     ]
errorq5p=[    abs((y[i]-yq5p[i])/ y[i])     for i in range(len(y))]


R=np.loadtxt("corrZqMPSq5lay14.txt")
xq5b=R[:,2]
yq5b=R[:,3]
yq5b=[    abs(yq5b[i])     for i in range(len(yq5b))     ]
errorq5b=[    abs((y[i]-yq5b[i])/ y[i])     for i in range(len(y))]




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
yq4l8b=[    abs(yq4l8b[i])     for i in range(len(yq4l8b))     ]
errorq4l8b=[    abs((abs(y[i])-abs(yq4l8b[i]))/ y[i])     for i in range(len(y))]


R=np.loadtxt("corrZq4l12.txt")
xq4l12b=R[:,2]
yq4l12b=R[:,3]
yq4l12b=[    abs(yq4l12b[i])     for i in range(len(yq4l12b))     ]
errorq4l12b=[    abs((abs(y[i])-abs(yq4l12b[i]))/ y[i])     for i in range(len(y))]




R=np.loadtxt("corrZq9l6.txt")
xq9l6b=R[:,2]
yq9l6b=R[:,3]
yq9l6b=[    abs(yq4l12b[i])     for i in range(len(yq9l6b))     ]
errorq9l6b=[    abs((abs(y[i])-abs(yq9l6b[i]))/ y[i])     for i in range(len(y))]



R=np.loadtxt("corrZQMPSq4l20.txt")
xq4b=R[:,2]
yq4b=R[:,3]
errorq4b=[    abs((y[i]-yq4b[i])/ y[i])     for i in range(len(y))]
yq4b=[    abs(yq4b[i])     for i in range(len(yq4b))     ]





plt.figure(figsize=(8, 6))


#plt.loglog( x, y, '--', lw=2,color = '#204a87', label='DMRG')

#plt.plot( xD16, errorD16, '--', color = '#204a87',markersize=10, label=r'dMPS(DMRG), $D=16$')
#plt.plot( x, yq4b, '1', color = '#e30b69',markersize=22, label=r'$qMPS, q=4$')


errorl8p=[   0.9*errorl7p[i]     for i in range(len(errorl7p))] 

#plt.plot( x, yl4p, '-p', color = '#e30b69',markersize=10, label=r'$QC-l, \tau=4$')
#plt.plot( x, errorl6p, '-.D', color = '#f57900',markersize=11, label=r'QC-l, $\tau=6$')
#plt.plot( x, errorl7p, '--D', color = '#72cfbb',markersize=10, label=r'QC-l, $\tau=7$')
plt.plot( x, errorl8p, '--D', color = '#72cfbb',markersize=11, label=r'QC-l, $\tau=8$')
plt.plot( x, errorl12, '--s', color = '#e90ff5',markersize=11, label=r'QC-b, $\tau=12$')

plt.plot( xq4l5b, errorq4l5b, '-->', color = '#cf729d',markersize=11, label=r'qMPS, $q=4, \tau=5$')
plt.plot( xq4l12b, errorq4l12b, '-->', color = '#a40000',markersize=11, label=r'qMPS, $q=4, \tau=12$')

plt.plot( xq2M, errorq2M, '--o', color = '#5c3566',markersize=11, label=r'qMERA, $q=2, \tau=6$')
plt.plot( xq3M, errorq3M, '--h', color = '#ce5c00',markersize=11, label=r'qMERA, $q=3,  \tau=10$')



#plt.loglog( x, yq5p,'3', color = '#cf729d',markersize=15, label=r'$qMPS, n_q=5$')
#plt.loglog( x, yq5b,'2',markersize=15, color = '#c74294', label=r'$qMPS, n_q=5$')


plt.yscale("log")

#plt.title('qmps')
plt.ylabel(r'$\delta C(r)$',fontsize=20)
plt.xlabel(r'$r$',fontsize=20)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.xlim([0.,24])
plt.ylim([ 1.e-4, 1.e0])


plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(loc="lower right", prop={'size': 18})


plt.grid(True)
plt.savefig('correrror.pdf')
plt.clf()
