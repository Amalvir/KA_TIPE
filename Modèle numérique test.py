import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

## Objectifs:
# Obtenir la position, et le volume immergé en fonction de l'angle téta

# Obtenir le MSIT en fonction des positions relatives des centres de carène et de gravité et du Volume:
""" On modélise notre engin flottant par un pavé de profondeur b, de hauteur h, de largeur l """

h = 9.5
# b=
l = 34
robj = 0.8*10**3   # Masse volumique de l'objet en kg/m^3
rofl = 10**3   # Masse volumique du fluide kg/m^3
# Vp est le volume du pavé
# Vp=l*b*h
# I est le moment quadratique
# H rayon du cercle que forme les centres de carène en fonction de téta
# i hauteur immergée de l'objet d'après archimède:
i = robj/rofl*h
# V=i*l*b  #V volume de la partie imergée
# I=b*h**3/12 
# H=I/V
# a=  #a distance entre G et C le centre de carène ce qu'est relou à calculer
# MSIT=rofl*(I-V*a) 
# 
# def affichage_de_la_situation_initiale3D(teta):
#     X=[0,l,l,0,0,l,l,0]
#     Y=[0,0,b,b,0,0,b,b]
#     Z=[0,0,0,0,h,h,h,h]
#     


#On reste dans le plan
def affichage_la_situation_initiale2D():
    X = [-l/2, l/2, l/2, -l/2, -l/2]  #ABCDA
    Z = [h - i, h - i, -i, -i, h - i]
    EAU = [-2*l, 2*l]
    NIV_EAU = [0, 0]

    plt.plot([0], [(h - 2*i)/2], 'o', label='G')
    plt.plot([0], [-i/2], 'o', label='C')
    plt.plot([0], [0], 'o', label='O')
    plt.plot(X, Z)
    plt.plot(EAU, NIV_EAU, label='eau')
    plt.legend()


# def affichage(teta):
#     phi = np.arctan((h-i)/2)
#     d = ((l/2)**2 + (h-i)**2)**1/2
#     C = [d*np.cos(phi + teta), d*np.sin(phi + teta)]
#     D = [-C[0] + l*np.cos(teta), C[1] - l*np.sin(teta)]
#     A = [D[0] - h*np.cos(teta), h*np.sin(teta) - D[1]]
#     B = [np.cos(teta)*h + C[0], np.sin(teta)*h - C[1]]
#     X = [A[0], B[0], C[0], D[0], A[0]]
#     Y = [A[1], B[1], C[1], D[1], A[1]]
#     plt.plot(X,Y)
#     plt.show()

def rotation(teta):
    rDC = ((l/2)**2 + (h-i)**2)**(1/2) #on avait oublié les parenthèses dans l'exposant
    rAB = (i**2 + (l/2)**2)**(1/2)

    phiDC = np.arctan((h - i) /(l/2))  #faute de frappe
    print("phiDC", phiDC)
    phiAB = np.arctan((i/(l/2)))
    print("phiAB", phiAB)
    # Pour vérif' on ajoute les positions intiales avec complexes
    C0 = rDC*np.exp(1j*phiDC)
    D0 = rDC*np.exp(1j*(-phiDC+np.pi))
    B0 = rAB*np.exp(1j*(-phiAB))
    A0 = rAB*np.exp(1j*(np.pi + phiAB))

    Cj = rDC*np.exp(1j*(phiDC + teta))
    Dj = rDC*np.exp(1j*(np.pi - phiDC + teta))
    Bj = rAB*np.exp(1j*(teta - phiAB))
    Aj = rAB*np.exp(1j*(np.pi + phiAB + teta))

    plt.plot([Cj.real, Dj.real, Aj.real, Bj.real, Cj.real], [Cj.imag, Dj.imag, Aj.imag, Bj.imag, Cj.imag])
    plt.plot([C0.real, D0.real, A0.real, B0.real, C0.real], [C0.imag, D0.imag, A0.imag, B0.imag, C0.imag])
<<<<<<< HEAD
    
=======
    print(rDC)
>>>>>>> 8dd47de5fc14151f5a0d7c748e15524b678d84ac




<<<<<<< HEAD
# On décide de tourner autour du point O, projeté de G sur NIV_EAU
# def affichage_la_situationB(teta):
#     Bx = np.cos(teta)*(l/2)+l/2+i*np.sin(teta)
#     Bz = (l/2)*np.sin(teta)+(Bx-l)*np.tan(teta)  # ya une erreur recalculer abs et ordonnées de ABCDA
#     plt.plot([Bx], [Bz], 'o', label='B')
#     plt.show()
# 
# rotation(0)
# affichage_la_situation_initiale2D()
# plt.show()
=======
def affichage_la_situationB(teta):
    Bx = np.cos(teta)*(l/2)+l/2+i*np.sin(teta)
    Bz = (l/2)*np.sin(teta)+(Bx-l)*np.tan(teta)  # ya une erreur recalculer abs et ordonnées de ABCDA
    plt.plot([Bx], [Bz], 'o', label='B')
    plt.show()

plt.figure(figsize=[10, 10])
plt.axis([-20, 20, -20, 20])
rotation(np.pi/6)
affichage_la_situation_initiale2D()

# On décide de tourner autour du point O, projeté de G sur NIV_EAU
plt.savefig("Magnifique.pdf")
>>>>>>> 8dd47de5fc14151f5a0d7c748e15524b678d84ac
