# La où ca affiche des trucs

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Calcul import *    # pylint: disable=unused-wildcard-import


def init():
    """Plot les conditions initiales"""

    X, Z = reel(rotation(0, affichage=True))
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
    X, Z = reel(rotation(teta))
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
    X, Z = reel(rotation(0, affichage=True))
    rect = ax.plot(X, Z)[0]
    plot1 = ax.plot([], [], 'o', color="orange")[0]
    plot2 = ax.plot([], [], 'o', color="orange")[0]
    buoyency = ax.plot([], [], 'o-', color="green")[0]
    grav = ax.plot([], [], 'o', color="red")[0]
    init()

    def animate(agl):
        root = racines(agl)
        Rot = rotation(agl, affichage=False)
        X, Z = reel(tri(Rot))   # On convertie et on trie les points
        xg, zg = center_of_mass(X, Z)   # Point G
        X, Z = reel(tri(Rot + root))
        Xb, Zb = immerg((X, Z), agl)   # Point immergé
        Xbb, Zbb = center_of_buoyancy(Xb, Zb, agl)  # Centre de buyocency
        X, Z = reel(tri(rotation(agl, True)))
        rect.set_data(X, Z)
        plot1.set_data(root[0], [0])
        plot2.set_data(root[1], [0])
        buoyency.set_data(Xbb, Zbb)
        grav.set_data(xg, zg)

        return rect, plot1, plot2, buoyency, grav

    ani = animation.FuncAnimation(fig, animate, frames=teta, blit=True, interval=15)
    plt.show()


def non_anim(teta):
    """Fonction qui génère la rotation"""
    init()
    X, Z = reel(rotation(teta, affichage=True))
    plt.plot(X, Z)
    xg, zg = center_of_mass(X, Z)   # Point G
    plt.plot(xg, zg, 'o', color='red')
    root = racines(teta)
    X.extend(root)
    Z.extend([0]*len(root))
    Xg, Zg = immerg((X, Z), teta)
    plt.plot(Xg[0], Zg[0], 'o', color='green')   # Le centre de gravité des points immergé
    points(teta)
    for j in root:
        plt.plot(j, [0], 'o', color="orange")


def courbe_de_stabilite_statique():
    X=np.linspace(0,2*np.pi) #abscisses de la courbe
    Y=[GZ(x) for x in X ]
    plt.plot(X,Y,label='courbe de stabilitée satique')
    #plt.plot([1,1],[5,8],"--")# verticale pour tangeante à l'origine
    #plt.plot([0,1.5],[distance_Gmetacentre(),distance_Gmetacentre()],"--")  #Gm pour construire la tangeante à l'origine
    plt.plot([0,1],[0,distance_Gmetacentre()],label='tangeante à l origine') #tangeante à l'origine
     #MSIT
    plt.plot(X,Z)
    plt.xlabel('teta')
    plt.ylabel('GZ')
    plt.legend()
    plt.show()

#pb de positionnement du métacentre 
def Aff_metacentre_teta():
    X=np.linspace(0,np.pi/10)
    Y=[distance_Gmetacentre(x) for x in X ]
    plt.plot(X,Y)
    plt.show()