# La où on exécute des trucs (dans les autres y'a que des fonctions)

from Affichage import *

# affichage(np.linspace(0, 2*np.pi, 500))
# affichage(np.pi/120)


# print(aire_immerg(np.pi/120))
X = np.linspace(0, 4*np.pi, 400)
Y = [aire_immerg(teta) for teta in X]
plt.plot(X, Y)
plt.show()
