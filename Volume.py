from Initial import *

# Le volume immergé est constant
# On le calcule par la poussée d'Archimède

# Constante
h = 9.5
l = 34
robj = 0.8*10**3   # Masse volumique de l'objet en kg/m^3
rofl = 10**3   # Masse volumique du fluide kg/m^3

def i(teta):
    """Renvoie la hauteur immergé en fonction du rho objet, fluide, hauteur et teta"""
    i0 = robj/rofl*h
    A0 = l*i0

    # On récupère les coords des points
    X, Y = rotation(teta)
    # # systeme d'équation
    # 0 = np.tan(teta)*x + b
    # Y[0] = tan(teta)*X[0] + b

    # Matrices :
    A = np.array([[np.tan(teta), 1], [0, 1]])
    B = np.array([0, Y[0] - np.tan(teta)*X[0]])
    S = np.linalg.solve(A, B)
    # S = [x, b] coordonné du point ou ca touche l'eau : [x, 0]
    print(S)
    return S[0]


