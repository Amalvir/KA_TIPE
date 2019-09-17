# La où ca affiche des trucs

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Calcul import *


def init():
    """Plot les conditions initiales"""

    X, Z = rotation(0)
    EAU = [-2*l, 2*l]
    NIV_EAU = [0, 0]

    plt.plot([0], [(h - 2*i)/2], 'o', color='red', label='G')
    plt.plot([0], [-i/2], 'o', color='green', label='C')
    plt.plot([0], [0], 'o', color='gray', label='O')
    plt.plot(X, Z)
    plt.plot(EAU, NIV_EAU, label='eau')
    points(0)


def points(teta):
    """Affiche les points ABCD ainsi que leurs noms en fonction de teta"""
    X, Z = rotation(teta)
    P = ['C', 'D', 'A', 'B']
    for j in range(len(P)):
        plt.plot(X[j], Z[j], 'ob')
        plt.annotate(P[j], xy=(X[j], Z[j]))


def affichage(teta):
    """Crée l'animation si teta est une liste. Crée le rectangle si c'est juste un angle"""
    fig, ax = plt.subplots(1, figsize=[7, 7])
    ax.axis([-20, 20, -20, 20])
    if isinstance(teta, list) or isinstance(teta, np.ndarray):
        anim(teta, fig, ax)
    else:
        non_anim(teta)
        plt.legend()
    plt.show()


def anim(teta, fig, ax):
    """Fonction qui génère l'animation"""
    X, Z = rotation(0)
    rect = ax.plot(X, Z)[0]
    plot1 = ax.plot([], [], 'o', color="orange")[0]
    plot2 = ax.plot([], [], 'o', color="orange")[0]
    buoyency = ax.plot([], [], 'o', color="green")[0]
    grav = ax.plot([], [], 'o', color="red")[0]
    init()

    def animate(agl):
        X, Z = rotation(agl)
        xg, zg = center_of_mass(X, Z)
        root = racines(agl)
        X.extend(root)
        Z.extend([0]*len(root))
        Xb, Zb = immerg(X, Z)
        X, Z = rotation(agl)

        rect.set_data(X, Z)
        plot1.set_data(root[0], [0])
        plot2.set_data(root[1], [0])
        buoyency.set_data(Xb[0], Zb[0])
        grav.set_data(xg, zg)

        return rect, plot1, plot2, buoyency, grav

    print(1)
    ani = animation.FuncAnimation(fig, animate, frames=teta, blit=True, interval=15)
    plt.show()


def non_anim(teta):
    """Fonction qui génère la rotation"""
    init()
    X, Z = rotation(teta)
    plt.plot(X, Z)
    xg, zg = center_of_mass(X, Z)
    plt.plot(xg, zg, 'o', color='red')
    root = racines(teta)
    X.extend(root)
    Z.extend([0]*len(root))
    Xg, Zg = immerg(X, Z)
    print(Zg)
    plt.plot(Xg[0], Zg[0], 'o', color='green')   # Le centre de gravité des points immergés
    # print(Zg)
    plt.plot(Xg[0], Zg[0], 'o', color='green')   # Le centre de gravité des points immergé
    points(teta)
    for j in root:
        plt.plot(j, [0], 'o', color="orange")


def courbe_de_stabilite_statique():
    X=np.linspace(0,np.pi/2) #abscisses de la courbe
    Y=[GZ(x) for x in X ]
    plt.plot(X,Y,label='courbe de stabilitée satique')
    plt.plot([1,1],[5,8],"--") # verticale pou tangeante à l'origine
    plt.xlabel('teta')
    plt.ylabel('GZ')
    plt.legend()
    plt.show()


    