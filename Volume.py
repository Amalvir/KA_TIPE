# Le volume immergé est constant
# On le calcule par la poussée d'Archimède

# Constante

def i(rhobj, rhofl, h):
    """Renvoie la hauteur immergé en fonction du rho objet, fluide et de la hauteur"""
    return rhobj/rhofl*h

