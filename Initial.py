# La où on exécute des trucs (dans les autres y'a que des fonctions)

from Affichage import *

affichage(np.linspace(0, np.pi/3, 200))
# affichage(np.pi/40)


# print(aire_immerg(np.pi/120))
X = np.linspace(0, np.pi/40, 100)
Y = [aire_immerg(teta) for teta in X]
plt.plot(X, Y)
plt.show()
