from Calcul import *
from Affichage import *

##Dimensions utilisées:

#b= 11.2*1e-2  # profondeur sert pas dans l'animation mais pour le MSIT
#h=3.5*1e-2 # Hauteur
#l=11.2*1e-2  # Largeur
#robj=813  #avec le poids de la maquette:357g

dis=0.05
##Tracé de la courbe stabilité statique pour la maquette:
AVS=courbe_de_stabilite_statique()

#AVS=1.5707963267948966 rad 90°
#Même courbe de stabilité statique que la vrai éolienne: maquette plutôt bonne
####Calcul de l'angle de tangage dû au poids de l'éolienne

poids=[0.01,0.02,0.03]

def fonc_angle(poids):
    angle=[]
    for x in poids:
        def angle_recherche(teta):
            """fonction utile pour trouver l'angle auquel penche le flotteur lorsqu'on lui met un poids"""
            return Mt(teta)-x*dis
        a=so.newton(angle_recherche,np.pi/60)
        angle.append(a*180/np.pi)
    return angle

angle=fonc_angle(poids)
#print(angle)
#[2.4182325236516626, 4.962098456024369, 10.580594717658995]
# 
# plt.plot(poids,angle)
# plt.show()

