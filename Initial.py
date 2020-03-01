# La où on exécute des trucs (dans les autres y'a que des fonctions)

from Affichage import * # pylint: disable=unused-wildcard-import


# affichage(np.linspace(-np.pi/3, np.pi/3, 1000))
# affichage(-np.pi/4)
# teta = np.pi/40
# a = h/2


rect = Rectangle()
rect._rotation(-np.pi/3)
# print(rect.coords)
# rect._translation()
# plt.plot([-2*l, 2*l], [0,0])
# plt.plot(rect.X, rect.Z)

X = np.linspace(-17, 17, 34)
Y = [rect.f(x) for x in X]

plt.plot(X, Y)
plt.show()