##Objectifs:
#Obtenir la position, et le volume immergé en fonction de l'angle téta 

#Obtenir le MSIT en fonction des positions relatives des centres de carène et de gravité et du Volume:
""" On modélise notre engin flottant par un pavé de profondeur b, de hauteur h, de largeur l """
h=
b=
l=
robj=   #Masse volumique de l'objet
rofl=   #Masse volumique du fluide 
#Vp est le volume du pavé
Vp=l*b*h
#I est le moment quadratique 
#H rayon du cercle que forme les centres de carène en fonction de téta
#i hauteur immergée de l'objet d'après archimède:
i=(robj/rofl)*h
V=i*l*b  #V volume de la partie imergée
I=b*h**3/12
H=I/V
a=  #a distance entre G et C le centre de carène
MSIT=

def affichage_de_la_situation_initiale(teta):
    X=[0,l,l,0,0,l,l,0]
    Y=[0,0,b,b,0,0,b,b]
    Z=[]
    