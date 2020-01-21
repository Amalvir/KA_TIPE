# La où on exécute des trucs (dans les autres y'a que des fonctions)

from Affichage import * # pylint: disable=unused-wildcard-import


# affichage(np.linspace(0, np.pi/3, 500))
# affichage(np.pi/40)

# L = tri(rotation(np.pi/40))
teta = np.pi/40
X = np.linspace(0.1, h)
Y = [aire_immerg(teta, a=k) for k in X]
plt.plot(X, Y)
plt.show()

# print(aire_immerg(np.pi/120))
# X = np.linspace(0, np.pi/40, 100)
# Y = [aire_immerg(teta) for teta in X]
# plt.plot(X, Y)
# plt.show()

# Imposer Aire immergé constante. Rtourné le programme pour faire en fonction de aire immerg et de teta
