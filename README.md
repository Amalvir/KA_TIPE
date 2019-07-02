# KA_TIPE
Notre TIPE

## Objectifs:
* Obtenir la position, et le volume immergé en fonction de l'angle téta
* Obtenir le MSIT en fonction des positions relatives des centres de carène et de gravité et du Volume:
  On modélise notre engin flottant par un pavé de profondeur b, de hauteur h, de largeur l

h = 9.5 m

l = 34 m

### Masse volumique de l'objet en kg/m^3 :
* robj = 0.8*10**3

### Masse volumique du fluide kg/m^3 :
* rofl = 10**3

### Vp est le volume du pavé :
* Vp=l*b*h

### I est le moment quadratique
I=b*h**3/12
 
### H rayon du cercle que forme les centres de carène en fonction de téta
 
### i hauteur immergée de l'objet d'après archimède:
= robj/rofl*h
 
### V volume de la partie imergée
V=i*l*b  V volume de la partie imergée
 
H=I/V
 
### a distance entre G et C le centre de carène ce qu'est relou à calculer
a=  a distance entre G et C le centre de carène ce qu'est relou à calculer
 
MSIT=rofl*(I-V*a)
