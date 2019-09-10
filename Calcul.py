# La où y'a les calculs

import numpy as np

# Constante
b=34  #profondeur sert pas dans l'animation mais pour le MSIT
h = 9.5
l = 34
robj = 0.8*10**3   # Masse volumique de l'objet en kg/m^3
rofl = 10**3   # Masse volumique du fluide kg/m^3
i = robj/rofl*h   # i hauteur immergée de l'objet d'après archimède:


def rotation(teta):
    """Renvoie les listes des X et Z des points ABCD du rectangle ayant fait une rotation teta"""
    rDC = ((l/2)**2 + (h-i)**2)**(1/2)
    rAB = (i**2 + (l/2)**2)**(1/2)

    phiDC = np.arctan((h - i)/(l/2))
    phiAB = np.arctan((i/(l/2)))

    Cj = rDC*np.exp(1j*(phiDC + teta))
    Dj = rDC*np.exp(1j*(np.pi - phiDC + teta))
    Bj = rAB*np.exp(1j*(teta - phiAB))
    Aj = rAB*np.exp(1j*(np.pi + phiAB + teta))

    return [Cj.real, Dj.real, Aj.real, Bj.real, Cj.real], [Cj.imag, Dj.imag, Aj.imag, Bj.imag, Cj.imag]


def racines(teta):
    """Renvoie les coords en x des points de contacts avec l'eau en fonction de teta"""

    sol = []
    # On récupère les coordonnés des points
    X, Y = rotation(teta)

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


def immerg(X, Z):
    """D'une liste de points, renvoie ceux qui sont immergés avec le centre de gravité en premier"""
    Z1 = []
    X1 = []
    for a in range(len(Z)):
        if Z[a] <= 1e-2:    # On est en python alors on prend une petite valeur plutôt que 0
            Z1.append(Z[a])
            X1.append(X[a])
    x1, z1 = center_of_buoyancy(X1, Z1)
    X1.insert(0, x1)
    Z1.insert(0, z1)
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
    return sum(X[:-1])/len(X[:-1]), sum(Z[:-1])/len(Z[:-1])


def center_of_buoyancy(X, Z):
    """Renvoie les coords du centre de buoyency"""
    return sum(X)/len(X), sum(Z)/len(Z)

def distance_entreGC(teta):
    X,Z=rotation(teta)
    x,z=immerg(X,Z)
    return np.sqrt((center_of_mass(x,z)[0]-center_of_mass(X,Z)[0])**2+(center_of_mass(x,z)[1]-center_of_mass(X,Z)[1])**2)
    
    
I=b*h**3/12   # I est le moment quadratique

# MSIT=rofl*(I-Vc*a) 
#def fMSIT(teta):
 # return rofl*(I-aire_immerg(teta)*distance_entreGC(teta)

def aire_immerg(teta):
    """Calcul l'aire de la partie immergée en fonction de l'angle teta"""
    A, B = rotation(teta)
    R = racines(teta)

    C = A[1:]
    D = B[1:]
    E, F = immerg(C, D)
    X = E[1:]
    Y = F[1:]

    X.insert(0, R[0])
    Y.insert(0, 0)
    X.append(R[1])
    Y.append(0)
    print(X, Y)

    s = 0
    for i in range(len(X)-1):
        s = s + X[i]*Y[i+1] - X[i+1]*Y[i]
        # On doit trouver 323
    return 1/2*s



