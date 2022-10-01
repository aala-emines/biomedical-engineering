import numpy as np
import matplotlib.pyplot as plt
n = int(input())
m= int(input())
D = int(input())
L= int(input())
delta_x= L/n
delta_y = D/m
c=int(input())
#denombrement
kdict = {}
j,i,times= 0,0,0
while j < m:
     if i == n:
        j+=1
        i = 0
        times +=1
     else:
        kdict[(i,j)] = i+j + times*(n-1)
        i+=1
def k(i,j):#pour simplifier
    return kdict[(i,j)]
a = np.zeros((n*m)**2).reshape(n*m,n*m) #la matrice A
for i in range(1,n-1):
    for j in range(1,m-1):
        a[k(i,j),k(i+1,j)] = 1/delta_x**2
        a[k(i,j), k(i-1,j)] = 1/delta_x**2
        a[k(i,j), k(i,j)] =-2/delta_x**2 -2/delta_y**2
        a[k(i,j),k(i,j+1)] = 1/delta_y**2
        a[k(i,j),k(i,j-1)] = 1/delta_y**2
b = np.array([c for i in range((n*m))]) #la matrice B
#conditions de dirichlet
for j in(0,m-1):
    for i in range(n):
        a[k(i,j),k(i,j)] = 1
        b[k(i,j)] = 0
#conditions de neumann
for j in range(m):
        a[k(0,j), k(0,j)] =1
        a[k(0,j), k(0, j)+1] = 1
        b[k(0,j)] = 0
for j in range(m):
    a[k(n-1, j), k(n-1, j)] = 1
    a[k(n-1, j), k(n-1,j) - 1] = 1
    b[k(n-1, j)] = 0
for i in range(n):
    a[k(i, 0), k(i, 0)] = 1
    a[k(i, 0), k(i, 0) + 1] = -1
    b[k(i, 0)] = 0
for i in range(n):
    a[k(i, m-1), k(i,m-1)] = 1
    a[k(i, m-1), k(i, m-1) - 1] = -1
    b[k(i, m-1)] = 0
#resolution
X = np.linalg.solve(a,b) #vecteur contenant les images de chaque point de la grille
#representation
row = np.linspace(0,n,n)
column = np.linspace(0,m,m)
X = np.reshape(X,(n,m))# needs to be discussed, the numerotation may be different...
h = plt.contourf(row, column, X)
plt.axis('scaled')
plt.colorbar()
plt.show()