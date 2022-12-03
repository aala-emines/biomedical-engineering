import sys
import numpy as np
import matplotlib.pyplot as plt

L = 6e-2
N = 60
l = 2e-2
M = 60
P1 = 10300
P2 = 10000
eta = 0.000018

DX = L / N
DY = l / M

D = ((N + 1) * (M + 1))
R = []
Resistance = []
Eca_Type = []
alpha1=3
for r in range(1,(N-alpha1-3)):
    R.append(r)
    A = np.zeros((3 * D, 3 * D))
    B = np.zeros((3 * D, 1))

    alpha = alpha1
    LX = r
    LY = 6

    beta = alpha1
    LX1 = r
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
    for i in range(1, LX):
        for j in range(0, LY):
            Interieur_Def.append(alpha + i + j * (N + 1))

    for i in range(1, LY1):
        Cote_Gauche.append(beta - i * (N + 1))
        Cote_Droit.append(beta + LX1 - i * (N + 1))
    for i in range(1, LX1):
        Bord_Bas.append(beta - LY1 * (N + 1) + i)
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
    HL3 = l2/n2
    ListeDebi = []
    for j in range(0, n1):
        S = 0
        for i in range(0, n2):
            Y = np.sqrt(ux[i * n1 + j] ** 2 + uy[i * n1 + j] ** 2)
            S = S + Y
        S = S * HL3
        moyenne_debit += S
        ListeDebi.append(S)

    moyenne_debit = moyenne_debit / n1

    Resistance.append((p1-p2)/moyenne_debit)

    ecart_type = 0
    for i in range(len(ListeDebi)):
        ecart_type = ecart_type + ((ListeDebi[i] - (moyenne_debit)) ** 2)

    ecart_type = np.sqrt((1 / n1) * ecart_type)

    Eca_Type.append(ecart_type)
    print("hello")
print(Eca_Type)

arrety = 6
arretx = 3
eta = 0.000018
l1 = 6e-2
n1 = N
l2 = 2e-2
n2 = M
n1+=1
n2+=1
L3=[] #liste des arretx
R3=[] #liste des resistance
ox= 3
R4 = []
def Rh(l,x):
    #l est la longueur de la portion concerne
    #x est le diametre de la portion concerne
    return 12*eta*l/(x**3)

def diam(l,y,n) : #convertion de subdivision au longueur
    return y*l/(n-1)

while arretx<(n1-ox-3) :
    L3.append(arretx*(l2/n2)) #rajoute a la liste la hauteur de la deformation
    Rt = 0 #initialisation de la resistance totale
    Rt += Rh(diam(l1,arretx,n1),diam(l2,n2-2*arrety,n2)) #on rajoute la resistance de la deformation
    Rt += Rh(diam(l1,ox,n1),l2) #on ajoute la resistance a gauche
    Rt += Rh(diam(l1,n1-1 - ox - arretx,n1),l2) # on ajoute la resistance a droite
    R3.append(Rt)
    arretx += 1
print(len(L3))

plt.plot(np.array(R)*(l2/n2)*10*(3),Resistance,label = "R_simulé")
plt.xlabel('Longeur de la deformation en mm')
plt.ylabel('Resistance par simulation')
plt.title("Largeur fixée de la deformation ="+str(diam(l2,arrety,n2)*10*(3)) + "mm, graphe obtenue pour l1,l2=  " +str(l1*10**(2))+" cm, "+str(l2*10**(2))+"cm")
plt.legend()
plt.grid()
plt.subplots()
plt.plot(np.array(L3)*10*(3),R3,label = "R_poiseuille")
plt.xlabel('Longeur de la deformation en mm')
plt.ylabel('Resistance par approximation de Poiseuille')
plt.title("Longeur fixée de la deformation ="+str(diam(l2,arrety,n2)*10*(3)) + "mm, graphe obtenue pour l1,l2=  " + str(l1*10**(2))+" cm, "+str(l2*10**(2))+" cm")
plt.legend()
plt.grid()
plt.subplots()
quotient = [Resistance[i] / R3[i] for i in range(len(R3))]
plt.plot(np.array(L3)*10*(3),np.array(quotient), label = "R_simulée/ R_poiseuille")
plt.show()
