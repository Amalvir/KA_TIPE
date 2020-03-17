from Calcul import *
from Affichage import *

##Dimensions utilisées:

#b= 11.2*1e-2  # profondeur sert pas dans l'animation mais pour le MSIT
#h=3.5*1e-2 # Hauteur
#l=11.2*1e-2  # Largeur
#robj=813  #avec le poids de la maquette:357g

dis=0.05
##Tracé de la courbe stabilité statique pour la maquette:
#AVS=courbe_de_stabilite_statique()

#AVS=

####Calcul de l'angle de tangage dû au poids de l'éolienne

poids=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5]

def fonc_angle(poids):
    angle=[]
    for x in poids:
        p=x
        def angle_recherche(teta):
            """fonction utile pour trouver l'angle auquel penche le flotteur lorsqu'on lui met un poids"""
            return Mt(teta)-p*dis
        a=so.newton(angle_recherche,np.pi/60)
        angle.append(a*180/np.pi)
    return angle

angle=fonc_angle(poids)

plt.plot(poids,angle)
plt.show()

