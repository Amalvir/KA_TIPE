# La où ca affiche des trucs

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Calcul import *


def init():
    """Plot les conditions initiales"""

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

    plt.show()


def anim(teta, fig, ax):
    """Fonction qui génère l'animation"""
    X, Z = rotation(0)
    rect = ax.plot(X, Z)[0]
    plot1 = ax.plot([], [], 'o', color="orange")[0]
    plot2 = ax.plot([], [], 'o', color="orange")[0]
    init()

    def animate(agl):
        X, Z = rotation(agl)
        rect.set_data(X, Z)
        root = racines(agl)
        plot1.set_data(root[0], [0])
        plot2.set_data(root[1], [0])

        return rect, plot1, plot2

    print(1)
    ani = animation.FuncAnimation(fig, animate, frames=teta, blit=True, interval=15)
    plt.show()


def non_anim(teta):
    """Fonction qui génère la rotation"""
    plt.figure(figsize=[7, 7])
    plt.axis([-20, 20, -20, 20])
    init()
    X, Z = rotation(teta)
    plt.plot(X, Z)
    points(teta)
    root = racines(teta)
    for j in root:
        plt.plot(j, [0], 'o', color="orange")
    print(2)