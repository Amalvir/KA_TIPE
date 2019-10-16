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

    @property
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
    
    

Eoe = eolienne(0).sommets
print(Eoe)