import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
# import Volume as V

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
    """Plot les conditions initiales"""
    # X = [-l/2, l/2, l/2, -l/2, -l/2]  #ABCDA
    # Z = [h - i, h - i, -i, -i, h - i]
    X, Z = rotation(0)
    EAU = [-2*l, 2*l]
    NIV_EAU = [0, 0]

    plt.plot([0], [(h - 2*i)/2], 'o', label='G')
    plt.plot([0], [-i/2], 'o', label='C')
    plt.plot([0], [0], 'o', label='O')
    plt.plot(X, Z)
    plt.plot(EAU, NIV_EAU, label='eau')
    points(0)
    plt.legend()




def rotation(teta):
    """Renvoie la liste des X et Z pour plot la rotation"""
    rDC = ((l/2)**2 + (h-i)**2)**(1/2)
    rAB = (i**2 + (l/2)**2)**(1/2)

    phiDC = np.arctan((h - i)/(l/2))
    # print("phiDC", phiDC)
    phiAB = np.arctan((i/(l/2)))
    # print("phiAB", phiAB)

    # C0 = rDC*np.exp(1j*phiDC)
    # D0 = rDC*np.exp(1j*(-phiDC+np.pi))
    # B0 = rAB*np.exp(1j*(-phiAB))
    # A0 = rAB*np.exp(1j*(np.pi + phiAB))

    Cj = rDC*np.exp(1j*(phiDC + teta))
    Dj = rDC*np.exp(1j*(np.pi - phiDC + teta))
    Bj = rAB*np.exp(1j*(teta - phiAB))
    Aj = rAB*np.exp(1j*(np.pi + phiAB + teta))

    return [Cj.real, Dj.real, Aj.real, Bj.real, Cj.real], [Cj.imag, Dj.imag, Aj.imag, Bj.imag, Cj.imag]


def racines(teta):
    """Renvoie les points de constacts avec l'eau en fonction de teta"""

    sol = []
    # On récupère les coords des points
    X, Y = rotation(teta)
    for i in range(len(Y) - 1):
        if Y[i]*Y[i+1] < 0:
            coeff = (Y[i+1] - Y[i])/(X[i+1] - X[i])
            A = np.array([[coeff, 1], [0, 1]])
            B = np.array([0, Y[i] - coeff*X[i]])
            S = np.linalg.solve(A, B)
            sol.append(S[0])

    # # systeme d'équation
    # 0 = np.tan(teta)*x + b
    # Y[0] - tan(teta)*X[0] = b

    # Matrices :
    # A = np.array([[np.tan(teta), 1], [0, 1]])
    # B = np.array([0, Y[0] - np.tan(teta)*X[0]])
    # S = np.linalg.solve(A, B)
    # S = [x, b] coordonné du point ou ca touche l'eau : [x, 0]
    # print(sol)
    return sol


def points(teta):
    """Affiche les points ABCD en fonction de teta"""
    X, Z = rotation(teta)
    P = ['C', 'D', 'A', 'B']
    for i in range(len(P)):
        plt.plot(X[i], Z[i], 'ob')
        plt.annotate(P[i], xy=(X[i], Z[i]))


def affichage(teta):
    fig = plt.figure(figsize=[7, 7])
    camera = Camera(fig)
    plt.axis([-20, 20, -20, 20])
    affichage_la_situation_initiale2D()

    camera.snap()
    if isinstance(teta, list) or isinstance(teta, np.ndarray):
        affichage_la_situation_initiale2D()
        for agl in teta:
            X, Z = rotation(agl)
            plt.plot(X, Z)
            points(agl)
            root = racines(agl)
            for j in root:
                plt.plot(j, [0], 'o', color="orange")
            camera.snap()

        anim = camera.animate()
        plt.show()
    # else:
    #     affichage([teta])



    # X, Z = rotation(teta)
    # plt.plot(X, Z)
    # points(teta)
    # root = racines(teta)
    # for i in root:
    #     plt.plot(i, [0], 'o', color="orange")

    # affichage_la_situation_initiale2D()
    # On décide de tourner autour du point O, projeté de G sur NIV_EAU
    # plt.savefig("Magnifique.pdf")


L = np.linspace(0, np.pi/6, 50)


affichage(L)
