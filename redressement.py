import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def critique(t, A, B, C, f, phi):
    return A + B*t + C*np.sin(2*np.pi*f*t + phi)

def amorti(t, A, B, C, tau, f, phi):
    return A + B*t + C*np.exp(-t/tau)*np.sin(2*np.pi*f*t + phi)

for i in range(3):
    data = np.genfromtxt("{}.csv".format(i), delimiter=";", skip_header=1)
    X = np.linspace(data[0,0], data[-1,0], 1000)
    popt, pcov = curve_fit(amorti, data[:,0], data[:,1], p0=[0, 0, 0, 1, 2, 0])
    A, B, C, tau, f, phi = popt


    data[:,1] -= B*data[:,0] + A
    print(i, ":", "{} + {}*t + {}*sin({}*t + {})".format(A, B, C, tau, f, phi))
    np.savetxt("sortie{}.csv".format(i), data, delimiter=";")

    plt.plot(X, amorti(X, A, B, C, tau, f, phi), label="modèle")
    plt.plot(data[:,0], data[:,1], 'o', label="Point originaux")

    plt.legend()
    plt.show()
    print("f=",f)

