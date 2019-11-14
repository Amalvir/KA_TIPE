# La où y'a les calculs

import numpy as np

# Constante
b = 34  # profondeur sert pas dans l'animation mais pour le MSIT
h = 9.5
l = 34
robj = 0.7*10**3   # Masse volumique de l'objet en kg/m^3
rofl = 10**3   # Masse volumique du fluide kg/m^3
i = robj/rofl*h   # i hauteur immergée de l'objet d'après archimède:
I = b*h**3/12   # I est le moment quadratique


def rotation(teta, affichage=False, a=i):
    """Renvoie les listes des X et Z des points ABCD du rectangle ayant fait une rotation teta"""
    rDC = ((l/2)**2 + (h-a)**2)**(1/2)
    rAB = (a**2 + (l/2)**2)**(1/2)

    phiDC = np.arctan((h - a)/(l/2))
    phiAB = np.arctan((a/(l/2)))

    Cj = rDC*np.exp(1j*(phiDC + teta))
    Dj = rDC*np.exp(1j*(np.pi - phiDC + teta))
    Bj = rAB*np.exp(1j*(teta - phiAB))
    Aj = rAB*np.exp(1j*(np.pi + phiAB + teta))

    if affichage:
        return [Cj, Dj, Aj, Bj, Cj]
    else:
        return [Cj, Dj, Aj, Bj]

def tri(L):
    """Trie la liste en fonction des arguments"""
    # L = np.angle(A)
    for k in range(1, len(L)):
        temp = L[k]
        j = k
    while j > 0 and np.angle(temp) < np.angle(L[j-1]):
        L[j] = L[j-1]
        j -= 1
        L[j] = temp
    return L
   


def reel(L):
    """D'une liste de complexe, renvoie les coords X et Y"""
    X = []
    Y = []
    for i in L:
        X.append(i.real)
        Y.append(i.imag)
    return (X, Y)

def racines(teta):
    """Renvoie les coords en x des points de contacts avec l'eau en fonction de teta"""

    sol = []
    # On récupère les coordonnés des points
    X, Y = reel(rotation(teta, affichage=True))

    for j in range(len(Y) - 1):
        if Y[j]*Y[j+1] < 0:
            # On test si les coordonnées Y sont 2 à 2 de même signes
            # Calcul du coeff directeur de la droite
            coeff = (Y[j+1] - Y[j])/(X[j+1] - X[j])

            # Système d'équations : (inconnues : x, b)
            # coeff*x + b = 0
            # 0*x + b = Y[j] - coeff*X[j]
            A = np.array([[coeff, 1], [0, 1]])      # Matrices des variables
            B = np.array([0, Y[j] - coeff*X[j]])    # Matrices des constantes
            S = np.linalg.solve(A, B)   # Pivot de Gauss
            sol.append(S[0])    # On a besoin que de x donc on append que S[0]
    return sol


def immerg(coords, teta):
    """D'une liste de points, renvoie ceux qui sont immergés avec le centre de gravité en premier"""
    X, Z = coords
    Z1 = []
    X1 = []
    for a in range(len(Z)):
        if Z[a] <= 1e-5:    # On est en python alors on prend une petite valeur plutôt que 0
            X1.append(X[a])
            Z1.append(Z[a])
    # x1, z1 = center_of_buoyancy(X1, Z1, teta)
    # X1.insert(0, x1)
    # Z1.insert(0, z1)
    # print('X1', X1, 'Z1', Z1, '\n')
    return X1, Z1


def emerg(X, Z):
    """D'une liste de points, renvoie ceux qui sont émergés."""
    Z1 = []
    X1 = []
    for a in range(len(Z)):
        if Z[a] >= -1e-2:  # On est en python alors on prend une petite valeur plutôt que 0
            Z1.append(Z[a])
            X1.append(X[a])
    return X1, Z1


def center_of_mass(X, Z):
    """Renvoie les coords du centre de gravité"""
    return sum(X)/len(X), sum(Z)/len(Z)


def center_of_buoyancy(X, Z, teta):
    """Renvoie les coords du centre de buoyency"""
    X += X[:1]
    Z += Z[:1]
    s = 0
    t = 0
    for i in range(0, len(X)-1):
        s += (X[i] + X[i+1])*(X[i]*Z[i+1]-X[i+1]*Z[i])
        t += (Z[i] + Z[i+1])*(X[i]*Z[i+1]-X[i+1]*Z[i])
    return 1/(6*aire_immerg(teta))*s, 1/(6*aire_immerg(teta))*t

def distance_entreGC(teta):
    X ,Z = reel(rotation(teta))
    x, z = immerg((X, Z), teta)
    return np.sqrt((center_of_mass(x,z)[0]-center_of_mass(X,Z)[0])**2 + (center_of_mass(x,z)[1]-center_of_mass(X,Z)[1])**2)
    

# MSIT=rofl*(I-Vc*a) 
def fMSIT(teta):
    return rofl*(I-(aire_immerg(teta)*b*distance_entreGC(teta)))


def aire_immerg(teta, a=i):
    """Calcul l'aire de la partie immergée en fonction de l'angle teta"""
    Rot = tri(rotation(teta, affichage=False, a=a) + racines(teta))
    X, Y = immerg(reel(Rot), teta)

    X += X[:1]
    Y += Y[:1]

    s = 0
    for i in range(len(X)-1):
        s = s + X[i]*Y[i+1] - X[i+1]*Y[i]
        print(1/2*s)
        # On doit trouver 323
    return 1/2*s

def GZ(teta):
    """Revoit le bras de levier entre le centre de masse en position initiale et le centre de masse en fonction de teta"""
    A,B=rotation(0)
    C,D=rotation(teta)
    Z=center_of_mass(A,B)
    G=center_of_mass(C,D)
    return np.sqrt((Z[0]-Z[1])**2+(G[0]-G[1])**2)


def metacentre():
    teta=np.pi/65
    A,B=rotation(0)
    C,D=rotation(teta)
    G=center_of_mass(A,B)
    Gp=center_of_mass(C,D)
    c1,c2=immerg(A,B)
    c3,c4=immerg(C,D)
    C=[c1[0],c2[0]]
    Cp=[c3[0],c4[0]]


