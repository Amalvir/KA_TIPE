from Calcul import *
from Affichage import *
import scipy.optimize as so
import scipy.integrate as si



##Dimensions utilisées pour la vérification:
poids=133000 #en kilos
dis=27       #en m

#le flotteur fait 4360 tonnes
#dim du flotteur 34*34*9.5 à l'ext
#robj=397.0132
#le poids de l'éolienne: 133tonnnes


##Tracé de la courbe de stabilité statique pour l'éolienne en taille réelle
AVS=courbe_de_stabilite_statique()
#AVS=1.5707963267948966=pi/2
#Angle of vanishing stability plutô élévé: sécurité; il faut que le flotteur soit à 90° avant de tendre vers la position retournée comme position stable 
#bateau à voile Giro AVS 128° (lui il bouge pas l'éolienne)

##Calcul de l'angle de tangage dû au poids de l'éolienne
def angle_recherche(teta):
    """fonction utile pour trouver l'angle auquel penche le flotteur lorsqu'on lui met un poids"""
    return Mt(teta)-poids*dis

angle=so.newton(angle_recherche,np.pi/200)


# X=np.linspace(0,np.pi/60)
# Z=[poids*dis,poids*dis]
# Y=[Mt(teta) for teta in X]
# plt.plot([angle],[poids*dis],'o')
# plt.plot(X,Y)
# plt.plot([0,np.pi/60],Z)
# plt.show()


#pour poids=133000 et dis de 27m
#angle=2.091996172244318 deg
#Conclusion ça penche un peu quand même mais bon, modélisation sommaire

##Calcul de l'energie nécéssaire au chavirement:
def fonc_energie(teta):
    return abs(GZ(teta))
    
Energie,err=si.quad(fonc_energie,0,np.pi) #il y a encore des recherches à faire je sais pas l'unité/quels bornes exactement...


