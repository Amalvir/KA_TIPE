# La où ca affiche des trucs

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Calcul import *    # pylint: disable=unused-wildcard-import
import scipy.optimize as so


def init(rect):
    """Met en place la position initiale du rectangle"""
    rect.rot(0)
    X, Z = rect.X, rect.Z
    EAU = [-2*l, 2*l]
    NIV_EAU = [0, 0]
    xg, zg = rect.center_of_mass
    xb, zb = rect.center_of_buoyancy

    plt.plot([xg], [zg], 'o', color='red', label='G')
    plt.annotate('G', (xg, zg))
    plt.plot([xb], [zb], 'o', color='green', label='C')
    plt.annotate('C', (xb, zb))
    plt.plot([0], [0], 'o', color='gray', label='O')
    plt.plot(X, Z)
    plt.plot(EAU, NIV_EAU, label='eau')
    points(rect)


def points(rect):
    """Affiche les points ABCD ainsi que leurs noms de l'objet Rectangle"""
    X, Z = rect.X, rect.Z
    P = ['C', 'D', 'A', 'B']
    for j in range(len(P)):
        plt.plot(X[j], Z[j], 'ob')
        plt.annotate(P[j], xy=(X[j], Z[j]))


def affichage(teta):
    """Affiche une animation si teta est une liste et une image si teta est un nombre."""
    fig, ax = plt.subplots(1, figsize=[7, 7])
    ax.axis([-2*l, 2*l, -2*l, 2*l])
    rect = Rectangle()
    if isinstance(teta, list) or isinstance(teta, np.ndarray):
        anim(teta, rect, fig, ax)
    else:
        non_anim(rect, teta)
        plt.legend()
    plt.show()


def anim(teta, rectangle, fig, ax):
    """Fonction qui génère l'animation"""
    X, Z = rectangle.X, rectangle.Z
    rect = ax.plot(X, Z)[0]
    racines = ax.plot([-l/2, l/2], [0, 0], 'o', color="orange")[0]
    buoyency = ax.plot([], [], 'o-', color="green")[0]
    grav = ax.plot([], [], 'o', color="red")[0]
    init(rectangle)

    def animate(agl):
        rectangle.rot(agl)
        root = rectangle.rac
        X, Z = rectangle.X, rectangle.Z
        xg, zg = rectangle.center_of_mass   # Point G
        Xbb, Zbb = rectangle.center_of_buoyancy  # Centre de buoyency
        rect.set_data(X, Z)
        racines.set_data(root[:,0], root[:,1])
        buoyency.set_data(Xbb, Zbb)
        grav.set_data(xg, zg)
        return rect, racines, buoyency, grav

    ani = animation.FuncAnimation(fig, animate, frames=teta, blit=True, interval=15)
    # ani.save("animation.mp4")
    plt.show()


def non_anim(rect, teta):
    """Fonction qui affiche une image du rectange pivoté de teta"""
    init(rect)
    rect.rot(teta)
    X, Z = rect.X, rect.Z
    plt.plot(X, Z)
    xg, zg = rect.center_of_mass   # Point G
    plt.plot(xg, zg, 'o', color='red')
    root = rect.rac
    plt.plot(root[:,0], root[:,1], 'o', color="yellow")
    Xg, Zg = rect.center_of_buoyancy
    plt.plot([Xg], [Zg], 'o', color='green')   # Le centre de gravité des points immergé
    points(rect)


def courbe_de_stabilite_statique():
    X=np.linspace(0,np.pi) #abscisses de la courbe
    Y=[GZ(x) for x in X ]
    MSIT=[fMSIT(teta) for teta in X]
    AVS=so.newton(GZ,1)
    plt.plot(X,Y,label='courbe de stabilitée satique')
    #plt.plot([1,1],[0,12],"--")# verticale pour tangeante à l'origine
    #plt.plot([0,1.5],[distance_Gmetacentre(0.001),distance_Gmetacentre(0.001)],"--") #Gm pour construire la tangeante à l origine
    plt.plot([0,1],[0,distance_Gmetacentre(0.001)],label='tangeante à l origine') #tangeante à l'origine
    plt.plot([0,np.pi],[0,0])
    #plt.plot(X,MSIT,label='Module de stabilité initial transversal')
    plt.title('courbe de stabilité statique')
    plt.plot(X,Y)
    plt.plot([AVS],[0],"o",label='AVS') #Point AVS
    plt.xlabel('teta')
    plt.ylabel('GZ')
    plt.legend()
    plt.show()
    return AVS

#pb de positionnement du métacentre 
def Aff_metacentre_teta():
    X=np.linspace(0,np.pi/10)
    Y=[distance_Gmetacentre(x) for x in X ]
    plt.plot(X,Y)
    plt.show()
    
