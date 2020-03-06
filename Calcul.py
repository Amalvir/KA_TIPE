# La où y'a les calculs

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sc
import scipy.optimize as sc

# Constante
b = 11.2*10**(-2)  # profondeur sert pas dans l'animation mais pour le MSIT
h = 3.5*10**(-2) # Hauteur
l = 11.2*10**(-2)  # Largeur
robj = 0.7*10**3   # Masse volumique de l'objet en kg/m^3
rofl = 10**3   # Masse volumique du fluide kg/m^3
i = robj/rofl*h   # i hauteur immergée de l'objet d'après archimède:
I = l*b**3/12   # I est le moment quadratique du pavé lorsqu'il ne bouge pas
A = h*l*robj/rofl # Aire immergée

## Définition de la classe Rectangle

class Rectangle:
    """Notre objet rectangle"""

    def __init__(self):
        """Ici c'est les 'conditions inistiales'"""
        self.aff = np.array([[l/2, h/2],    # Le rectangle
                            [-l/2, h/2],
                            [-l/2, -h/2],
                            [l/2, -h/2]])

        self.rac = np.array([[-l/2, 0],     # Les racines
                             [l/2, 0]])

    def _rotation(self, teta):
        """Ne pas utiliser. Fais tourner rectangle d'un angle teta."""
        teta = -teta    # Inversion de l'angle sinon ça tourne dans le mauvais sens (je sais pas pk)
        Rot = np.array([[np.cos(teta), -np.sin(teta)],  # Matrice de rotation 2x2
                        [np.sin(teta), np.cos(teta)]])
        for i in range(self.aff.shape[0]):
            self.aff[i] = np.dot(self.aff[i], Rot)  # Application de la matrice de rotation à chaque points
        self._set_racines()     # Redéfinition des racines après la racines

    def __f(self, x):
        """Ne pas utiliser. Sert à la dichotomie de _translation. Permet de trouver x tel que
        si on translate le rectangle de x, l'aire immergée = A"""
        self.aff[:,1] += x  # Translation de x
        self._set_racines() # Redéfinition des racines
        y = self.aire_immerg - A
        self.aff[:,1] -= x  # Retour à la position précédente
        self._set_racines() # Redéfinition des racines
        return y
    
    def _translation(self):
        """Ne pas utiliser. Ajuste la hauteur du rectangle pour que aire immérgée = A"""
        a, b = -l/2, l/2
        # Dichotomie on cherche le 0 de la fonction __f
        while abs(b-a) > 0.01:
            c = (a+b)/2
            if self.__f(a)*self.__f(c) <= 0:
                b = c
            else:
                a = c
        self.aff[:,1] += c  # Tranlation
        self._set_racines() # Définition des racines
    
    def _set_racines(self):
        """Ne pas utiliser. Met dans l'attribut rac les coordonnées des points de coupure avec l'eau.
        None si solide complètement émergé."""
        # On récupère les coordonnés des points
        X, Z = self.X, self.Z
        i = 0   
        self.rac = np.array([[-l/2, 0], # Sinon erreur lors de l'assignation plus tard
                            [l/2, 0]])

        for j in range(len(Z) - 1):
            if Z[j]*Z[j+1] < 0: # On test si les coordonnées Z sont 2 à 2 de même signes
                if X[j+1] == X[j]:  # Si les points sont alignés verticalement
                    self.rac[i] = [X[j], 0]
                    i += 1
                else:
                    # Calcul du coeff directeur de la droite
                    coeff = (Z[j+1] - Z[j])/(X[j+1] - X[j])

                    # Système d'équations : (inconnues : x, b)
                    # coeff*x + b = 0
                    # 0*x + b = Y[j] - coeff*X[j]
                    A = np.array([[coeff, 1], [0, 1]])      # Matrices des variables
                    B = np.array([0, Z[j] - coeff*X[j]])    # Matrices des constantes
                    S = np.linalg.solve(A, B)   # Pivot de Gauss
                    self.rac[i] = [S[0], 0]
                    i += 1
        if i == 0:  # Si aucune racines
            self.rac = None

    def rot(self, teta):
        """Utiliser. Effectue une rotation du rectangle avec translation."""
        self.__init__()
        self._rotation(teta)
        self._translation()

    @property
    def coords(self):
        """Renvoie un array avec les coordonnées de chaque point du rectangle triées dans le 
        sens antihoraire avec les racines"""
        if type(self.rac) == type(None):    # Si pas de racines il n'y a que self.aff
            return self.aff
        else:
            coords = np.zeros((6,2))
            i = 0
            j = 0
            while i < 3:
                coords[i+j] = self.aff[i]
                if self.aff[i,1]*self.aff[i+1,1] < 0:
                    coords[i+j+1] = self.rac[j]
                    j += 1
                i += 1
            coords[i+j] = self.aff[i]
            if j == 1:
                coords[i+j+1] = self.rac[j]
            return coords
    
    @property
    def X(self):
        """Renvoie une liste des abscisses pour afficher le rectangle"""
        return list(self.aff[:,0]) + [self.aff[0,0]]
    
    @property
    def Z(self):
        """Renvoie une liste des ordonnées pour afficher le rectangle"""
        return list(self.aff[:,1]) + [self.aff[0,1]]


    @property
    def pol_immerg(self):
        """Renvoie un array avec les coordonnées des points immergés"""
        coords = self.coords
        longueur = coords.shape[0]
        pol = []
        for a in range(longueur):
            if coords[a,1] <= 1e-5:    # On est en python alors on prend une petite valeur plutôt que 0
                pol.append(coords[a])
        return np.array(pol)
    
    @property
    def aire_immerg(self):
        """Renvoie l'aire de la surface immergée"""
        pol = self.pol_immerg
        if pol.shape == (0,):
            return 0
        else:
            return aire(pol)
    
    @property
    def center_of_mass(self):
        """Renvoie un tuple qui correspond au centre de gravité du rectangle de coordonnée (X, Z)"""
        X = self.aff[:,0]
        Z = self.aff[:,1]
        return sum(X)/len(X), sum(Z)/len(Z)
    
    @property
    def center_of_buoyancy(self):
        """Renvoie un tuple qui correspond au centre de buoyancy du rectangle de coordonnée (X, Z)"""
        immerg = self.pol_immerg
        aire = self.aire_immerg
        if immerg.shape == (0,):
            return 0, 0     # Cas qui ne sera jamais visble quand on affiche
        X = list(immerg[:,0])
        Z = list(immerg[:,1])
        X += X[:1]
        Z += Z[:1]
        s = 0
        t = 0
        # Formule de calcul de centre gravités d'un polynome simple quelconque
        for k in range(0, len(X)-1):
            s += (X[k] + X[k+1])*(X[k]*Z[k+1]-X[k+1]*Z[k])
            t += (Z[k] + Z[k+1])*(X[k]*Z[k+1]-X[k+1]*Z[k])
        return 1/(6*aire)*s, 1/(6*aire)*t

## Fonction utile à la classe Rectangle

def aire(pol):
    """Calcul l'aire d'un polynome quelconque dont les points sont triés dans le sens antihoraire"""
    X = list(pol[:,0])
    Z = list(pol[:,1])
    X += X[:1]
    Z += Z[:1]
    s = 0
    for k in range(len(X)-1):
        s += X[k]*Z[k+1] - X[k+1]*Z[k]
    return 1/2*s

## Fonction d'exploitation

def distance_entreGC(teta):
    """renvoie la distance entre le centre de poussé et le centre de gravité"""
    R=Rectangle()
    R.rot(teta)
    return np.sqrt((R.center_of_buoyancy[0]-R.center_of_mass[0])**2 + (R.center_of_buoyancy[1]-R.center_of_mass[1])**2)
    
    
    
# MSIT=rofl*(I-Vc*a) 
def fMSIT(teta): #ATTENTION ne fonctionne que pour les petits angles
    """Module de stabilité initiale tranversale"""
    rect=Rectangle()
    return rofl*(I(teta)-(A*b*distance_entreGC(teta))) #G est au dessus de c donc on compte positivement la distance gc 

def I(teta):
    """moment quadratique de la surface de flottaison"""
    rect=Rectangle()
    rect.rot(teta)
    r1=rect.rac[1][0]
    r2=rect.rac[0][0]
    lp=abs(r1)+abs(r2)
    return (lp*b**3/12)

def f_verification(teta):
    r=Rectangle()
    r.rot(teta)
    
 
def GZ(teta):
    """Revoit le bras de levier entre le centre de masse  et le centre de carène en fonction de teta"""
    R=Rectangle()
    
    R.rot(teta)
    G=R.center_of_mass
    Z=R.center_of_buoyancy
    return (G[0]-Z[0])
    


   ##   def metacentre(teta):
    #     """renvoie la position du métacentre, cependant la position du métacentre n'est pas constante, on suppose que que c'est le cas pour le petits angles et on considère pi/65 comme un petit angle"""
    #     #Ateta=np.pi/65
    #     r=Rectangle()
    #     
    #     G=r.center_of_mass
    #     C=r.center_of_buoyancy
    #     
    #     r.rot(teta)
    # 
    #     Gp=r.center_of_mass
    #     Cp=r.center_of_buoyancy
    #     
    #     coeff_dir=(Gp[1]-Cp[1])/(Gp[0]-Cp[0])
    #     #On sait que la droite passant par G et C est d'équation x=0
    #     #On sait que la droite passant par Gp, Cp est d'equation y=coeff_dir*x+b
    #     #L'ordonnée à l'origine de (Gp,Cp) correspond à l'ordonnée du métacentre
    #     ym=Gp[1]-coeff_dir*Gp[0]
    #     
    #     return (ym)
    # 
    # 
        
def distance_Gmetacentre(teta):

    return GZ(teta)/(np.sin(teta))


 # def angle_de_deplacement(m,y):
 #    """renvoie l'angle de déplacement du flotteur pour le rajout d'un poids P disposé à l'abscisse y du centre"""
 #    return np.arctan(m*g*y/)
    
def Mt (teta): #•ATTENTION ne fonctionne que pour les petits angles !!!
    """Couple de redressement"""
    return fMSIT(teta)*np.sin(teta)
    

    
# X=np.linspace(0,np.pi/8)
# Y=[fMSIT(teta) for teta in X]
# plt.plot(X,Y)
# plt.show()
# # 
#         
