
from Calcul import *

##L'objectif est l'obtenir la réponse en position de notre à un angle soumis à la poussée d'archimède et à la pesanteur;
#On utilise le théorème du moment cinétique au centre de gravité; on fait l'hypothèse que l'on reste en statique des fluides car on est pour les petits angles;
#J*teta''=rofl*aire_immerg*g*GBx*sin(teta)

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

#{constantes

b = 11.2*10**(-2)  # profondeur sert pas dans l'animation mais pour le MSIT
h = 3.5*10**(-2)
l = 11.2*10**(-2)
robj = 0.8*10**3   # Masse volumique de l'objet en kg/m^3
g=9.81
angle_de_dep=-np.pi/65
aire_immergee=h*l*robj/rofl
J= robj*h*l*b*(h**2+l**2)/12 #moment d'inertie du pavé flottant

def tetapp(Y,t):
    teta,tetap=Y
    R=Rectangle()
    R.rot(teta)
    Projection_GCx=R.center_of_mass[0]-R.center_of_buoyancy[0]
    return np.array([tetap,-(rofl*aire_immergee*g*b*Projection_GCx)/J])
    #▓return np.array([tetap,-(rofl*aire_immergee*g*b*distance_entreGC(teta)*np.sin(teta))/J])
    
t=np.linspace(0,2,200)
Y0=np.array([angle_de_dep,0])
teta=odeint(tetapp,Y0,t)[:,0]
tetap=odeint(tetapp,Y0,t)[:,1]


plt.plot(t,teta,label='angle teta en fonction du temps' )
plt.title('Tracé de l angle teta en fonction du temps')
plt.xlabel('temps')
plt.ylabel('angle')
plt.legend()



import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


##Pour extraire les informations on 

def modele(t, C, f, phi):
    return C*np.sin(2*np.pi*f*t + phi)



X = t
popt, pcov = curve_fit(modele, t, teta, p0=[0.01, 2.5, 0])
C, f, phi = popt


print( "{}*sin({}*t + {})".format(C, f, phi))


plt.plot(X, modele(X, C, f, phi), label="modèle")


plt.legend()
plt.show()
print("f=",f)

# test=np.linspace(-np.pi,np.pi,100)
# test_Y=[distance_entreGC(x) for x in test]
# plt.plot(test,test_Y)
# plt.grid()
# plt.show()








