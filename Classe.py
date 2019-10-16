import numpy as np

b = 34  # profondeur sert pas dans l'animation mais pour le MSIT
h = 9.5
l = 34
robj = 0.7*10**3   # Masse volumique de l'objet en kg/m^3
rofl = 10**3   # Masse volumique du fluide kg/m^3
i = robj/rofl*h   # i hauteur immergée de l'objet d'après archimède:
I = b*h**3/12   # I est le moment quadratique

class eolienne:

    def __init__(self, teta):
        self.teta = teta
        sommets(self)

    
    def sommets(self):
        """Renvoie les listes des X et Z des points ABCD du rectangle ayant fait une rotation teta"""
        rDC = ((l/2)**2 + (h-i)**2)**(1/2)
        rAB = (i**2 + (l/2)**2)**(1/2)

        phiDC = np.arctan((h - i)/(l/2))
        phiAB = np.arctan((i/(l/2)))

        Cj = rDC*np.exp(1j*(phiDC + self.teta))
        Dj = rDC*np.exp(1j*(np.pi - phiDC + self.teta))
        Bj = rAB*np.exp(1j*(self.teta - phiAB))
        Aj = rAB*np.exp(1j*(np.pi + phiAB + self.teta))

        return [Cj, Dj, Aj, Bj]
    
    @property
    def racines(self):
        """Renvoie les coords en x des points de contacts avec l'eau en fonction de teta"""

    sol = []
    # On récupère les coordonnés des points
    X, Y = np.real(self.sommets), np.complex(self.sommets)
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
    

Eoe = eolienne(0).sommets
Eoe.argsort
print(Eoe)