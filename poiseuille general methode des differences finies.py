import numpy as np
import matplotlib.pyplot as plt
l1=int(input("entrer la longueur du tuyau "))
n1=int(input("entrez la subdivision de la longueur "))
l2=int(input("entrer la largeur du tuyau "))
n2=int(input("entrez la subdivision de la largeur "))
#les choisir tel que l1/n1 = l2/n2
eta = 1000#float(input("entrez la viscosité "))#lorsqu'il est trop grand le output est quasi parfait mais la pression prend de grandes valeurs, et lorsque
#eta est trop faible le output se degrade mais la pression suit un profil quasi lineaire
#il faut prendre n1 est n2 voisins de 100 si on prend l2 et l1 voisins de 1
p1 = int(input("entrez p1"))
p2 = int(input("entrez p2"))
n1+=1#nombre de points sur un segment horizentale
n2+=1#nombre de points sur un segment vertocale
a = np.zeros(((n1)*(n2)*3,(n1)*(n2)*3))#qui contient les coefficients
b=np.zeros(((n1)*(n2)*3,))
#l'inconnu x contient ux puis uy puis p
h1=l1/n1
h2=l2/n2
u0= [-20*x*(x-l2) for x in list(np.linspace(0,l2, n2+1))]#entrée poiseuille
#chaque 3 lignes de definissent un systeme a unique solution concernant un point particulier
for i in range(0,(n1)*(n2)*3,3):#remplissage par ligne
    # sur les parois:
    if 0<int(i/3) <n1 or int(i/3) > (n2-1)*(n1):#vérifier par un exemple n2=3,n1=2:
        a[i][int(i/3)] = 1
        a[i+1][int(i/3)+n1*n2] = 1
        a[i+2][int(i/3)+2*n1*n2] = 1
        b[i+2] = p1 + (p2-p1)*(int(i/3) % n1)/(n1-1)#je suppose que sur les parois la pression varie linéairement
    #sur l'entrée
    elif int(i/3)%n1==0:
        a[i][int(i/3)]=1
        a[i+1][int(i/3)+n1*n2]=1
        a[i+2][int(i/3) + 2* n1 * n2] = 1
        b[i] = u0[(int(i/3) // n1)]
        b[i+2] = p1
    #sur la sortie
    elif int(i/3)%n1==n1-1: #int(i/3)%n1 est exactement l'abscisse du point concerné
        a[i][int(i/3)] = 1
        a[i + 1][int(i/3) + n1 * n2] = 1
        a[i + 2][int(i/3) + 2* n1 * n2] = 1
        b[i] = u0[(int(i/3) // n1)]
        b[i + 2] = p2
    #maintenant on remplit la matrice pour l'intérieur de la grille
    else:
        #on se place en un point d'indice i/3
        #i refere a la 1ere ligne de notre systeme
        #i+1 refere a la 2eme ligne de notre systeme...
        #la colonne précise l'inconnu et en quel point on veut cet inconnu a la fois
        #on remplit la premiere ligne de notre systeme en i
        a[i][int(i/3) +1] = 1/(h1 ** 2)
        a[i][int(i/3) -1]= 1/(h1 ** 2)
        a[i][int(i/3)] = -2/(h1**2)-2/(h2**2)
        a[i][int(i/3) +n1] = 1/(h2**2)
        a[i][int(i/3) -n1] = 1/(h2**2)
        a[i][int(i/3) - 1 +2*n1*n2] = -1/(eta*h1) #lorsque le met +1 au lieu de -1 comme indice de la colonne dans a j'obtiens du chaos
        a[i][int(i/3) + 2*n1*n2] = 1/(eta*h1)
        #on passe a la 2eme ligne
        a[i+1][int(i/3) + 1+n1*n2] = 1/ (h1 ** 2)
        a[i+1][int(i/3) - 1+n1*n2] = 1 / (h1 ** 2)
        a[i+1][int(i/3)+n1*n2] = -2 / (h1 ** 2)-2 / (h2 ** 2)
        a[i+1][int(i/3)+n1 +n1*n2] = 1 / (h2 ** 2)
        a[i+1][int(i/3)-n1 +n1*n2] = 1 / (h2 ** 2)
        a[i+1][int(i/3) - n1 +2*n1*n2] = -1 / (eta * h2)  #lorsque je met +n1 au lieu de -n1 j'obtiens le chaos
        a[i+1][int(i/3) + 2*n1*n2] = 1 / (eta * h2)
        #puis a la troisieme ligne
        a[i+2][int(i/3)+1] = 1/h1
        a[i+2][int(i/3)] = -1/h1
        a[i+2][int(i/3)+n1+n1*n2] = 1/h2
        a[i+2][int(i/3)+n1*n2] = -1/h2
a = 10**(-30)*a
b =10**(-30)*b
#ces deux lignes servent a annuler ce qui est de l'ordre de 10**(-10) pour minimiser l'erreur
s= np.linalg.solve(a,b)
ux = s[0:n1*n2]
uy = s[n1*n2: 2*n1*n2]
p = s[2*n1*n2:]

show = np.reshape(p,(n1, n2))
for i in range(n1): print(show[i][:])
y = np.linspace(0,l1,n2)
x = np.linspace(0,l2,n1)
x,y = np.meshgrid(x,y)
M = np.hypot(ux, uy)
fig,ax = plt.subplots()
q = ax.quiver(x,y, ux, uy,ux,units='width', pivot='tail', width=0.022,
               scale=2*n1, scale_units= "x", headwidth = 1, headlength = 3)
ax.scatter(x, y, color='0.5', s=1)
fig.colorbar(q)
plt.show()