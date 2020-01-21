import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fonction(t, A, B, C, D, phi):
    return A + B*t + C*np.sin(D*t + phi)


for i in range(1):
    data = np.genfromtxt("{}.csv".format(i),delimiter=";",skip_header=1)
    X = np.linspace(data[0,0], data[-1,0], 1000)
    spl = interpolate.UnivariateSpline(data[:,0], data[:,1], s=0)
    popt, pcov = curve_fit(fonction, data[:,0], data[:,1], p0=(1,-1,1,12,1))
    A, B, C, D, phi = popt

    def redressement(red):
        for i in range(len(data[:,1])):
            red.append(data[i,1]-B*data[i,0])
    # red = []
    # redressement(red)
    data[:,1] -= B*data[:,0] + A

    np.savetxt("sortie{}.csv".format(i), data, delimiter=";")

    # plt.plot(X, fonction(X, A, B, C, D, phi), label="modèle")
    # plt.plot(data[:,0], data[:,1], 'o', label="Point originaux")
    # # plt.plot(data[:,0],red, label="Points redréssés")
    # plt.legend()
    # plt.show()
    # print("f=",D/(2*np.pi))
