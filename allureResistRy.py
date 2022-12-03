
import numpy as np
import matplotlib.pyplot as plt
L = 6e-2
N = 30
l = 2e-2
M = 30
P1 = 1 + 0.0162
P2 = 0
eta = 0.000018

DX = L / N
DY = l / M

D = ((N + 1) * (M + 1))
R = []
Resistance = []
Q=[]
Eca_Type = []
for r in range(1,(int(M/2)-1)):
    R.append(r)
    A = np.zeros((3 * D, 3 * D))
    B = np.zeros((3 * D, 1))

    alpha = 17
    LX = 6
    LY = r

    beta = 17
    LX1 = 6
    LY1 = r
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
            Y = np.sqrt(ux[i * n1 + j] ** 2 + uy[i * n1 + j] ** 2) * HL3
            S = S + Y
        moyenne_debit += S
        ListeDebi.append(S)
    moyenne_debit = moyenne_debit / n1
    Resistance.append((p1-p2)/moyenne_debit)
    Q.append(moyenne_debit)
    ecart_type = 0
    for i in range(len(ListeDebi)):
        ecart_type = ecart_type + ((ListeDebi[i] - (moyenne_debit)) ** 2)

    ecart_type = np.sqrt((1 / n1) * ecart_type)
    Eca_Type.append(ecart_type)


arrety = 0
arretx = 6

L3=[] #liste des arrety
R3=[] #liste des resistance
R4=[]
ox= 17
def Rh(l,x):
    #l est la longueur de la portion concerne
    #x est le diametre de la portion concerne
    return 12*eta*l/(x**3)

def diam(l,y,n) : #convertion de subdivision au longueur
    return y*l/(n-1)

while arrety<(n2/2-1) :
    L3.append(arrety) #rajoute a la liste la hauteur de la deformation
    Rt = 0 #initialisation de la resistance totale
    Rt += Rh(diam(l1,arretx,n1),diam(l2,n2-2*arrety,n2)) #on rajoute la resistance de la deformation
    Rt += Rh(diam(l1,ox,n1),l2) #on ajoute la resistance a gauche
    Rt += Rh(diam(l1,n1-1 - ox - arretx,n1),l2) # on ajoute la resistance a droite
    R3.append(Rt)
    R4.append(Rh(l1, diam(l2,n2-2*arrety,n2)))
    arrety += 1
np.array(R3)
np.array(L3)

R3 = R3[:len(Resistance)]
R4 = R4[:len(Resistance)]
L3 = L3[:len(Resistance)]
plt.plot(np.array(R)*(l2/n2)*10**(3),np.log(Resistance),label = "log(R_simulé)")
plt.xlabel('Hauteur de la deformation en mm')
plt.plot(np.array(L3) *(l2/n2)*10**(3),np.log(R3),label = "log(R_poiseuille)")
plt.xlabel('Hauteur de la deformation en mm')
plt.title("Largeur fixée de la deformation ="+str(diam(l2,arrety,n2)*10*(3)) + "mm, graphe obtenue pour l1,l2=  " + str(l1*10**(2))+" cm, "+str(l2*10**(2))+" cm")
plt.plot(np.array(L3) *(l2/n2)*10**(3),np.log(R4),label = "log(R_tube_centrale)")
plt.grid()
plt.legend()
fig = plt.figure()
ax = fig.add_subplot()

quotient1 = np.log([Resistance[i]/R3[i] for i in range(len(Resistance))])
quotient2 = np.log([Resistance[i]/R4[i] for i in range(len(R4))])
ax.plot(np.array(R)*(l2/n2), quotient1, label = "log(R_simulé / R_poiseuille)")
ax.plot(np.array(R)*(l2/n2), quotient2, label = " log(R_simulé / R_tube_centrale)")
plt.title("Largeur fixée de la deformation ="+str(diam(l2,arrety,n2)*10*(3)) + "mm, graphe obtenue pour l1,l2=  " + str(l1*10**(2))+" cm, "+str(l2*10**(2))+" cm")
ax.set_yticks(np.arange(0, max((max(quotient2), max(quotient1)))+1, 1.0)) # setting the tick
plt.grid()
plt.legend()
plt.show()
