# La où on exécute des trucs (dans les autres y'a que des fonctions)

from Affichage import * # pylint: disable=unused-wildcard-import


# affichage(np.linspace(np.pi/40, np.pi, 500))
# affichage(0)
# teta = np.pi/40
# a = h/2

# def f(x):
#     return A - aire_immerg(teta, a=x, ajust=0)


# X = np.linspace(-h, h)
# Y = [A - aire_immerg(teta, a=k, ajust=0) for k in X]
# plt.plot(X, Y)
# plt.show()

# print(aire_immerg(np.pi/120))
# X = np.linspace(0, np.pi/40, 100)
# Y = [aire_immerg(teta) for teta in X]
# plt.plot(X, Y)
# plt.show()

rect = Rectangle()
print(rect.coords)