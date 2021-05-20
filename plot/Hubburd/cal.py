from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA


#U=1
#filling even

# e0=-0.8378
# e1=-1.0264
# e2=-0.3378


e0=-0.9916
e1=-1.0264
e2=-0.6582

print (e2-e1, e1-e0)
print (e2-e1+(e1-e0), 0.5)







#U=4
# e0=-0.748
# e1=-0.56610
# e2=1.251


e0=-0.80943
e1=-0.56610
e2=0.5239


print (e2-e1, e1-e0)
print (e2-e1+(e1-e0), 2.0)



#U=6
# e0=-0.7197
# e1=-0.414
# e2=2.2807

e0=-0.7459
e1=-0.414
e2=1.2541


print (e2-e1, e1-e0)
print (e2-e1+(e1-e0), 3.0)

