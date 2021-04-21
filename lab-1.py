import numpy as np
import matplotlib.pyplot as plt
import fun_y2
import scipy as sci
#z 3.
a1 = np.array([np.linspace(1,5,5),
np.linspace(5,1,5)])
a2 = np.array([[10],[10],[10],[10],[10]])
a3=np.zeros((3,2))
a4=np.linspace(2,2,3)
a5=np.linspace(2,2,3)
a6=np.linspace(-90,-70,3)

A=np.block([[a4],[a5]])
A=np.block([[A],[a6]])
A=np.block([a3,A])
A=np.block([[a1],[A]])
A=np.block([A,a2])


#print(A)
'''
 [[  1.   2.   3.   4.   5.  10.]
  [  5.   4.   3.   2.   1.  10.]
  [  0.   0.   2.   2.   2.  10.]
  [  0.   0.   2.   2.   2.  10.]
  [  0.   0. -90. -80. -70.  10.]]
'''
#z 4.
B=A[1,:]+A[3,:]
#print(B)
'''
[[ 5.,  4.,  5.,  4.,  3., 20.]]
'''
#z 5.
C=np.max(A,0)
#print(C)
'''
[[ 5.,  4.,  3.,  4.,  5., 10.]]
'''
#z 6.

D=np.delete(B,0,0)
D=np.delete(D,4,0)


'''
D=([4., 5., 4., 3.])
'''

#z 7.
D[D==4]=0
'''
D=([0., 5., 0., 3.])

'''
#z 8.

E= C[C > np.min(C)]
E= E[E < np.max(E)]  
'''
E=([5., 4., 4., 5.])
'''

#z 9.
wiersz_max=np.array([])
wiersz_min=np.array([])
wiersze_A=np.shape(A)
kolumny_A=np.shape(A)
for i in range(0,wiersze_A[0],1):
    for j in range(0,kolumny_A[1],1):
        if A[i,j]==np.max(A):
            wiersz_max=np.block([wiersz_max,i])
        else: 
            if A[i,j]==np.min(A):
                wiersz_min=np.block([wiersz_min,i])


#print("wiersz max")
#print(wiersz_max) 
#print("wiersz min")
#print(wiersz_min)
'''

wiersz max
[0. 1. 2. 3. 4.]
wiersz min
[4.]
'''
#z 10.
MM1=E@D
#print(MM1)

MT1=E*D
#print(MT1)

'''
MT1=35.0
MT2=[ 0. 20.  0. 15.]
'''
#z 11.
def mac_kwad():
    mac=np.random.randint(11,size=(3,3))
    slad=mac[0,0]+mac[1,1]+mac[2,2]
    return mac,slad
[mac,slad]=mac_kwad()
#print(mac)    
#print(slad)
'''
mac= ([[1, 4, 4],
       [9, 2, 7],
       [1, 9, 3]])
slad= 6
'''
#z 12.
def zeruj(mac):
    rozmiar=np.shape(mac)
    mac=mac*(1-np.eye(rozmiar[0],rozmiar[0]))
    mac=mac*(1-np.fliplr(np.eye(rozmiar[0], rozmiar[0])))
    return mac
mac=zeruj(mac)
'''
mac= ([[0., 1., 0.],
       [6., 0., 4.],
       [0., 6., 0.]])
'''
#z 13.
def parzyste(mac):
    suma=np.array([0,0,0])
    rozmiar=np.shape(mac)
    for i in range(0,rozmiar[0],1):
        if i%2==0:
            suma=suma+mac[i]
    return suma
suma=parzyste(mac)
'''
mac=([[0., 5., 0.],
       [9., 0., 3.],
       [0., 0., 0.]])
suma= array([0., 5., 0.])
'''
#z 14.
def fun_lambda(x):
    y1=np.cos(2*x)
    return y1

x=np.linspace(-10,10,201)
plt.plot(x,fun_lambda(x),color='r',dashes=[4,4])

#z 15.
wynik=np.array([])
for i in x:
    wynik=np.append(wynik,fun_y2.fun_y2(i))

plt.plot(x,wynik,'g')
    
#z 17
def fun_y3(x):
    y3=3*fun_lambda(x)+fun_y2.fun_y2(x)
    return y3
wynik1=np.array([])
for i in x:
    wynik1=np.append(wynik1,fun_y3(i))
plt.plot(x,wynik1)

#z 18
mac1=np.array([[10,5,1,7],
              [10,9,5,5],
              [1,6,7,3],
              [10,0,1,5]])
mac2=np.array([[34],
              [44],
              [25],
              [27]])
mac3=np.linalg.solve(mac1,mac2)
#print(mac3)
x=np.ones((4,1))
wsp_b=mac1@x
y=np.linalg.solve(mac1,wsp_b)
#print(y)
'''
mac3=[[2.]
      [1.]
      [2.]
      [1.]]
y=[[1.]
   [1.]
   [1.]
   [1.]]
'''
#z 19.

x=np.linspace(0,2*np.pi,1000000)

calka=np.sum(2*np.pi/1000000*np.sin(x))
#print(calka)
'''
calka=1.747760160067998e-15
'''



    








