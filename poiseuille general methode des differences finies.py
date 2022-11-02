import numpy as np
import matplotlib.pyplot as plt
l1=int(input("entrer la longueur du tuyau "))
n1=int(input("entrez la subdivision de la longueur "))
l2=int(input("entrer la largeur du tuyau "))
n2=int(input("entrez la subdivision de la largeur "))
eta = 0.00001#float(input("entrez la viscosité "))
p1 = int(input("entrez p1"))
p2 = int(input("entrez p2"))
n1+=1#nombre de points sur un segment horizentale
n2+=1#nombre de points sur un segment vertocale
a = np.zeros(((n1)*(n2)*3,(n1)*(n2)*3))#qui contient les coefficients
b=np.zeros(((n1)*(n2)*3,))
#l'inconnu x contient ux puis uy puis p
h1=l1/n1
h2=l2/n2
c = lambda x:x**2
def count(i,j): return n1*i + j
def Rcount(k): return (k//n1, k%n1)
def withinDistortedArea(n1,n2,rx,ry,ox,oy):
    inVout = {}
    for i in range(n2):
        for j in range(n1):
            inVout[count(i,j)] =c(i-oy)/c(ry) + c(j-ox)/c(rx) <= 1 or c(i-n2+1)/c(ry) + c(j-ox)/c(rx) <= 1 # true si le point k(i,j) se situe dans une des deux ellipes deformant le cylindre, c-a-d a l'exterieur du cylindre
    return inVout
rx=1
ry =2
ox = int(n1)/2
oy = 0
inVout = withinDistortedArea(n1,n2,rx,ry,ox,oy)
#chaque 3 lignes de definissent un systeme a unique solution concernant un point particulier
for i in range(0,(n1)*(n2)*3,3):#remplissage par ligne
    # sur l'entrée
    if int(i / 3) % n1 == 0:  # int(i/3)%n1 est exactement l'abscisse du point concerné
        a[i][int(i / 3)] = -1
        a[i][int(i / 3) + 1] = 1
        a[i][int(i / 3) + n1 - 2] = 1
        a[i][int(i / 3) + n1 - 1] = -1
        a[i + 1][int(i / 3) + n1 * n2] = -1
        a[i + 1][int(i / 3) + 1 + n1 * n2] = 1
        a[i + 1][int(i / 3) + n1 - 2 + n1 * n2] = 1
        a[i + 1][int(i / 3) + n1 - 1 + n1 * n2] = -1

        a[i + 2][int(i / 3) + 2 * n1 * n2] = 1
        b[i + 2] = p1
    # sur les parois:
    elif inVout[int(i / 3)] or int(i / 3) < n1 or int(i / 3) >= (n2 - 1) * (n1):  # vérifier par un exemple n2=3,n1=2:
        a[i][int(i / 3)] = 1
        a[i + 1][int(i / 3) + n1 * n2] = 1
        a[i + 2][int(i / 3) + 2 * n1 * n2] = 1
        b[i + 2] = p1 + (p2 - p1) * (int(i / 3) % n1) / (n1 - 1)  # je suppose que sur les parois la pression varie linéairement
    # sur la sortie
    elif int(i / 3) % n1 == n1 - 1:
        a[i][int(i / 3) + n1 * n2] = -1
        a[i][int(i / 3) + n1 + n1 * n2] = 1
        a[i][int(i / 3) - n1 + 1 + n1 * n2] = 1
        a[i][int(i / 3) + n1 - n1 + 1 + n1 * n2] = -1

        a[i + 1][int(i / 3)] = -1
        a[i + 1][int(i / 3) + n1] = 1
        a[i + 1][int(i / 3) - n1 + 1] = 1
        a[i + 1][int(i / 3) + n1 - n1 + 1] = -1

        a[i + 2][int(i / 3) + 2 * n1 * n2] = 1
        b[i + 2] = p2
    # maintenant on remplit la matrice pour l'intérieur de la grille
    else:
        # on se place en un point d'indice i/3
        # i refere a la 1ere ligne de notre systeme
        # i+1 refere a la 2eme ligne de notre systeme...
        # la colonne précise l'inconnu et en quel point on veut cet inconnu a la fois
        # on remplit la premiere ligne de notre systeme en i
        a[i][int(i / 3) + 1] = 1 / (h1 ** 2)
        a[i][int(i / 3) - 1] = 1 / (h1 ** 2)
        a[i][int(i / 3)] = -2 / (h1 ** 2) - 2 / (h2 ** 2)
        a[i][int(i / 3) + n1] = 1 / (h2 ** 2)
        a[i][int(i / 3) - n1] = 1 / (h2 ** 2)
        a[i][int(i / 3) + 1 + 2 * n1 * n2] = -1 / (eta * h1)
        a[i][int(i / 3) + 2 * n1 * n2] = 1 / (eta * h1)
        # on passe a la 2eme ligne
        a[i + 1][int(i / 3) + 1 + n1 * n2] = 1 / (h1 ** 2)
        a[i + 1][int(i / 3) - 1 + n1 * n2] = 1 / (h1 ** 2)
        a[i + 1][int(i / 3) + n1 * n2] = -2 / (h1 ** 2) - 2 / (h2 ** 2)
        a[i + 1][int(i / 3) + n1 + n1 * n2] = 1 / (h2 ** 2)
        a[i + 1][int(i / 3) - n1 + n1 * n2] = 1 / (h2 ** 2)
        a[i + 1][int(i / 3) + n1 + 2 * n1 * n2] = -1 / (eta * h2)
        a[i + 1][int(i / 3) + 2 * n1 * n2] = 1 / (eta * h2)
        # puis a la troisieme ligne
        a[i + 2][int(i / 3) + 1] = 1 / h1
        a[i + 2][int(i / 3)] = -1 / h1
        a[i + 2][int(i / 3) + n1 + n1 * n2] = 1 / h2
        a[i + 2][int(i / 3) + n1 * n2] = -1 / h2
s= np.linalg.solve(a,b)
ux = s[0:n1*n2]
uy = s[n1*n2: 2*n1*n2]
p = s[2*n1*n2:]

y = np.linspace(0,l1,n2)
x = np.linspace(0,l2,n1)
x,y = np.meshgrid(x,y)
M = np.hypot(ux, uy)
fig,ax = plt.subplots()
q = ax.quiver(x,y, ux, uy,(ux**2+uy**2)**(0.5),units='width', pivot='tail', width=0.022,
               scale=None, scale_units= "x", headwidth = 1, headlength = 3)
ax.scatter(x, y, color='0.5', s=1)
fig.colorbar(q)
plt.show()

# fig,ax = plt.subplots()
# row = np.linspace(0,n1,n1)
# column = np.linspace(0,n2,n2)
# np.reshape(column,(n2,1))
# ux = np.reshape(ux,(n2,n1))
# uy = np.reshape(uy,(n2,n1))
# X = (ux**2+uy**2)**(0.5)
# X = np.reshape(X,(n2,n1))#
# plt.contourf(row, column, X)
# plt.axis('scaled')
# plt.colorbar()
# plt.show()
