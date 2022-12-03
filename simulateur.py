import sys
import numpy as np
import matplotlib.pyplot as plt

'''
L=int(input("entrer la longueur du tuyau "))
N=int(input("entrez la subdivision de la longueur "))
l=int(input("entrer la largeur du tuyau "))
M=int(input("entrez la subdivision de la largeur "))
P1=int(input("entrez la pression d'entrée "))
P2=int(input("entrez la pression de sortie "))
eta=float(input("entrez la viscosité de votre fluide "))
'''
L = 6e-2
N = 10
l = 2e-2
M = 10
P1 =0.0162
P2 = 0
eta = 0.000018

DX = L / N
DY = l / M

D = ((N + 1) * (M + 1))
A = np.zeros((3 * D, 3 * D))
B = np.zeros((3 * D, 1))

alpha = 10
LX = 6
LY = 6

beta = 10
LX1 = 6
LY1 = 6
beta = beta + M * (N + 1)

Cote_Gauche = []
Cote_Droit = []
Bord_Haut = []
Bord_Bas = []
Interieur_Def = []

for i in range(1, LY):
    Cote_Gauche.append(alpha + i * (N + 1))
    Cote_Droit.append(alpha + LX + i * (N + 1))
for i in range(1, LX):
    Bord_Haut.append(alpha + LY * (N + 1) + i)

for i in range(1, LY1):
    Cote_Gauche.append(beta - i * (N + 1))
    Cote_Droit.append(beta + LX1 - i * (N + 1))
for i in range(1, LX1):
    Bord_Bas.append(beta - LY1 * (N + 1) + i)

for i in range(1, LX):
    for j in range(0, LY):
        Interieur_Def.append(alpha + i + j * (N + 1))

for i in range(1, LX1):
    for j in range(0, LY1):
        Interieur_Def.append(beta + i - j * (N + 1))

for i in range(0, D):
    if i % (N + 1) == 0:

        A[i][i] = 1
        A[i][i + N] = -1
        B[i][0] = 0
        # periodicite entre les bords gauche et droit du domaine Ux

        A[i + D][i + D] = 1
        A[i + D][i + D + N] = -1
        B[i + D][0] = 0
        # periodicite entre les bords gauche et droit du domaine Uy

        A[i + 2 * D][i + 2 * D] = 1
        B[i + 2 * D][0] = P1
        # pression sur le bord gauche du domaine

    elif i % (N + 1) == N:

        A[i][i] = 1 / DX
        A[i][i - 1] = -1 / DX
        A[i][i - N] = 1 / DX
        A[i][i - N + 1] = -1 / DX
        B[i][0] = 0
        # periodicite de la derivé entre les bords gauche et droit du domaine Ux

        A[i + D][i + D] = 1 / DX
        A[i + D][i + D - 1] = -1 / DX
        A[i + D][i + D - N] = 1 / DX
        A[i + D][i + D - N + 1] = -1 / DX
        B[i + D][0] = 0
        # periodicite de la derivé entre les bords gauche et droit du domaine Uy

        A[i + 2 * D][i + 2 * D] = 1
        B[i + 2 * D][0] = P2
        # pression sur le bord droit du domaine

    elif i == alpha:
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + 2 * D] = -(1 / DX) - (1 / DY)
        A[i + 2 * D][i + 2 * D + N + 1] = 1 / DY
        A[i + 2 * D][i + 2 * D - 1] = 1 / DX
        B[i + 2 * D][0] = 0
    elif i == (alpha + LX):
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + 2 * D] = -(1 / DX) - (1 / DY)
        A[i + 2 * D][i + 2 * D + N + 1] = 1 / DY
        A[i + 2 * D][i + 2 * D + 1] = 1 / DX
        B[i + 2 * D][0] = 0
    elif i == (alpha + LY * (N + 1)):
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i] = 1 / DX
        A[i + 2 * D][i - 1] = -1 / DX
        A[i + 2 * D][i + D] = -1 / DY
        A[i + 2 * D][i + D + N + 1] = 1 / DY
        B[i + 2 * D][0] = 0

    elif i == (alpha + LY * (N + 1) + LX):
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + 1] = 1 / DX
        A[i + 2 * D][i] = -1 / DX
        A[i + 2 * D][i + D] = -1 / DY
        A[i + 2 * D][i + D + N + 1] = 1 / DY
        B[i + 2 * D][0] = 0
    elif i == beta:
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + 2 * D] = -(1 / DX) - (1 / DY)
        A[i + 2 * D][i + 2 * D - N - 1] = 1 / DY
        A[i + 2 * D][i + 2 * D - 1] = 1 / DX
        B[i + 2 * D][0] = 0
    elif i == (beta + LX1):
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + 2 * D] = -(1 / DX) - (1 / DY)
        A[i + 2 * D][i + 2 * D - N - 1] = 1 / DY
        A[i + 2 * D][i + 2 * D + 1] = 1 / DX
        B[i + 2 * D][0] = 0
    elif i == (beta - LY1 * (N + 1)):
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i] = 1 / DX
        A[i + 2 * D][i - 1] = -1 / DX
        A[i + 2 * D][i + D - N - 1] = -1 / DY
        A[i + 2 * D][i + D] = 1 / DY
        B[i + 2 * D][0] = 0

    elif i == (beta - LY1 * (N + 1) + LX1):
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + 1] = 1 / DX
        A[i + 2 * D][i] = -1 / DX
        A[i + 2 * D][i + D - N - 1] = -1 / DY
        A[i + 2 * D][i + D] = 1 / DY
        B[i + 2 * D][0] = 0


    elif i in Cote_Gauche:
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i] = 1 / DX
        A[i + 2 * D][i - 1] = -1 / DX
        B[i + 2 * D][0] = 0
    elif i in Cote_Droit:
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + 1] = 1 / DX
        A[i + 2 * D][i] = -1 / DX
        B[i + 2 * D][0] = 0

    elif i in Bord_Haut:
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + D] = -1 / DY
        A[i + 2 * D][i + D + N + 1] = 1 / DY
        B[i + 2 * D][0] = 0

    elif i in Bord_Bas:
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + D - N - 1] = -1 / DY
        A[i + 2 * D][i + D] = 1 / DY
        B[i + 2 * D][0] = 0

    elif i in Interieur_Def:
        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0
        A[i + 2 * D][i + 2 * D] = 1
        B[i + 2 * D][0] = 0

    elif i <= N:

        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0

        A[i + 2 * D][i + D] = -1 / DY
        A[i + 2 * D][i + D + N + 1] = 1 / DY
        B[i + (2 * D)][0] = 0
        # derivee normale de la vitesse normale a la frontiere du domaine s’annule

    elif i >= (M * (N + 1)):

        A[i][i] = 1
        B[i][0] = 0
        # Ux = 0

        A[i + D][i + D] = 1
        B[i + D][0] = 0
        # Uy = 0

        A[i + 2 * D][i + D] = 1 / DY
        A[i + 2 * D][i + D - N - 1] = -1 / DY
        B[i + (2 * D)][0] = 0
        # derivee normale de la vitesse normale a la frontiere du domaine s’annule

    else:
        A[i][i - N - 1] = 1 / (DY * DY)
        A[i][i - 1] = 1 / (DX * DX)
        A[i][i] = - (2 / (DX * DX)) - (2 / (DY * DY))
        A[i][i + 1] = 1 / (DX * DX)
        A[i][i + N + 1] = 1 / (DY * DY)
        A[i][2 * D + i + 1] = -1 / (2 * eta * DX)
        A[i][2 * D + i - 1] = 1 / (2 * eta * DX)
        B[i][0] = 0
        # premiere equation laplacien(Ux) - (1/eta)*grad(Px)

        A[i + D][D + i - N - 1] = 1 / (DY * DY)
        A[i + D][D + i - 1] = 1 / (DX * DX)
        A[i + D][D + i] = - (2 / (DX * DX)) - (2 / (DY * DY))
        A[i + D][D + i + 1] = 1 / (DX * DX)
        A[i + D][D + i + N + 1] = 1 / (DY * DY)
        A[i + D][2 * D + i + N + 1] = -1 / (2 * eta * DY)
        A[i + D][2 * D + i - N - 1] = 1 / (2 * eta * DY)
        B[i + D][0] = 0
        # deuxieme equation laplacien(Uy) - (1/eta)*grad(Py)

        A[i + 2 * D][2 * D + i - N - 1] = 1 / (DY * DY)
        A[i + 2 * D][2 * D + i - 1] = 1 / (DX * DX)
        A[i + 2 * D][2 * D + i] = - (2 / (DX * DX)) - (2 / (DY * DY))
        A[i + 2 * D][2 * D + i + 1] = 1 / (DX * DX)
        A[i + 2 * D][2 * D + i + N + 1] = 1 / (DY * DY)
        B[i + 2 * D][0] = 0
        # div(u) = 0

n2 = (M + 1)
n1 = (N + 1)
l1 = L
l2 = l
p2 = P2
p1 = P1
s = np.linalg.solve(A, B)
ux = s[0:n1 * n2]
uy = s[n1 * n2: 2 * n1 * n2]
p = s[2 * n1 * n2:]
show = np.reshape(p, (n2, n1))
# for i in range(n2): print(show[i][:])
# p2=show[int(n2/2)][n1-1]

moyenne_debit = 0
HL3 = l2 / (n2)
ListeDebi = []
for j in range(0, n1):
    S = 0
    for i in range(0, n2):
        Y = np.sqrt(ux[i * n1 + j] ** 2 + uy[i * n1 + j] ** 2)
        S = S + Y
    S = S * HL3
    moyenne_debit += S
    ListeDebi.append(S)
    print(" á la pression", p[j], " Le debit vaut : ", S, " La résistance vaut ", (-p2 + p1) / S)
moyenne_debit = moyenne_debit / n1
print(" la moyenne des debit vaut ", moyenne_debit)
print(ListeDebi)
print(max(ListeDebi))
print(min(ListeDebi))
print("le plus grand ecart entre les debits est : ", (max(ListeDebi) - min(ListeDebi)))
ecart_type = 0
for i in range(len(ListeDebi)):
    ecart_type = ecart_type + ((ListeDebi[i] - (moyenne_debit)) ** 2)

ecart_type = np.sqrt((1 / n1) * ecart_type)
print("l'ecart type est : ", ecart_type)

y = np.linspace(0, l2, n2)
x = np.linspace(0, l1, n1)
x, y = np.meshgrid(x, y)
M = np.hypot(ux, uy)
fig, ax = plt.subplots()
q = ax.quiver(x, y, ux, uy, (ux ** 2 + uy ** 2) ** (0.5), units='width', pivot='tail', width=0.022,
              scale=None, scale_units="x", headwidth=1, headlength=3)
ax.scatter(x, y, color='0.5', s=1)
fig.colorbar(q)
plt.title("Écart-type = %f et debit-moyen = %f " % (ecart_type, moyenne_debit))

#

#
fig, ax = plt.subplots()
row = np.linspace(0, n1, n1)
column = np.linspace(0, n2, n2)
np.reshape(column, (n2, 1))
ux = np.reshape(ux, (n2, n1))
uy = np.reshape(uy, (n2, n1))
X = (ux ** 2 + uy ** 2) ** (0.5)
X = np.reshape(X, (n2, n1))  #
fig.set_figwidth(10)
fig.set_figheight(5)
plt.contourf(row, column, X)
plt.title("norme de vitesse pour l1,l2,n1,n2,p1,p2 = " + str((l1, l2, n1, n2, p1, p2)))
# plt.axis('scaled')
plt.colorbar()
fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(5)
plt.contourf(row, column, show,30,cmap = "magma")
plt.title("plot de pression, (debit,resistence )= (" + str(S) + "," + str((p2 + p1) / S))
# plt.axis('scaled')
plt.colorbar()
plt.show()
