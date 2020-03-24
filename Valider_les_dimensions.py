import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sc
import scipy.optimize as sc

####Valider les dimensions:

L=[i for i in range (30,100)]



def Valid_dim(L):
    """prends en argument un liste de larguer et renvois la liste correspondante d'AVS"""
    AVS=[] #on initialise la liste AVS
    Aire_flott=34*9.5 #aire du flotteur pour la garder constante
    for i in range (len(L)):
        b =L[i]  # profondeur sert pas dans l'animation mais pour le MSIT
        h =Aire_flott/L[i]# Hauteur
        l =L[i]  # Largeur

        ###On copie calcul pour pouvoir le modifier:
        g=9.81 #Accelération de la pesenteur
        robj =397.0132   # Masse volumique de l'objet en kg/m^3
        rofl = 1e3   # Masse volumique du fluide kg/m^3
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
                while abs(b-a) > 0.0001:
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
        
            
                    
        def GZ(teta):
            """Revoit le bras de levier entre le centre de masse  et le centre de carène en fonction de teta"""
            R=Rectangle()
            
            R.rot(teta)
            G=R.center_of_mass
            Z=R.center_of_buoyancy
            return (G[0]-Z[0])
            

        def fAVS():
            AVS=sc.newton(GZ,1)
            return AVS
            
        AVS.append(fAVS())
        
    return(AVS)



    


Y=Valid_dim(L)
plt.plot(L,Y,label='AVS en fonction de la largeur')
plt.legend()
plt.show()

        