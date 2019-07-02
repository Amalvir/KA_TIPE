import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np


h = 9.5
l = 34
robj = 0.8*10**3   # Masse volumique de l'objet en kg/m^3
rofl = 10**3   # Masse volumique du fluide kg/m^3
i = robj/rofl*h   # i hauteur immergée de l'objet d'après archimède:


#On reste dans le plan
def init():
    """Plot les conditions initiales"""

    X, Z = rotation(0)
    EAU = [-2*l, 2*l]
    NIV_EAU = [0, 0]

    plt.plot([0], [(h - 2*i)/2], 'o', label='G')
    plt.plot([0], [-i/2], 'o', label='C')
    plt.plot([0], [0], 'o', label='O')
    plt.plot(X, Z)
    plt.plot(EAU, NIV_EAU, label='eau')
    points(0)
    plt.legend()




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
    """Renvoie les points de contacts avec l'eau en fonction de teta"""

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


def points(teta):
    """Affiche les points ABCD ainsi que leurs noms en fonction de teta"""
    X, Z = rotation(teta)
    P = ['C', 'D', 'A', 'B']
    for j in range(len(P)):
        plt.plot(X[j], Z[j], 'ob')
        plt.annotate(P[j], xy=(X[j], Z[j]))


def affichage(teta):
    plt.figure(figsize=[7, 7])
    plt.axis([-20, 20, -20, 20])
    init()
    X, Z = rotation(teta)
    plt.plot(X, Z)
    points(teta)
    root = racines(teta)
    for i in root:
        plt.plot(i, [0], 'o', color="orange")

    plt.show()


affichage(np.pi/6)
