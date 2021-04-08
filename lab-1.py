#z 3.
import numpy as np
a1 = np.array([np.linspace(1,5,5),
np.linspace(5,1,5)])
a2 = np.array([[10],[10],[10],[10],[10]])
a3=np.zeros((1,3,2))
a4=np.linspace(2,2,3)
a5=np.linspace(2,2,3)
a6=np.linspace(-90,-70,3)

A=np.block([[a4],[a5]])
A=np.block([[A],[a6]])
A=np.block([[a3,A]])
A=np.block([[a1],[A]])
A=np.block([[A,a2]])


#print(A)
'''
[[[  1.   2.   3.   4.   5.  10.]
  [  5.   4.   3.   2.   1.  10.]
  [  0.   0.   2.   2.   2.  10.]
  [  0.   0.   2.   2.   2.  10.]
  [  0.   0. -90. -80. -70.  10.]]]
'''
#z 4.
B=A[:,1]+A[:,3]
#print(B)
'''
[[ 5.,  4.,  5.,  4.,  3., 20.]]
'''
#z 5.
C=np.max(A,1)
#print(C)
'''
[[ 5.,  4.,  3.,  4.,  5., 10.]]
'''
#z 6.

D=np.delete(B,0,1)
D=np.delete(D,4,1)
print(D)

'''
[[4., 5., 4., 3.]]
'''