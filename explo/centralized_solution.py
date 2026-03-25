import numpy as np
import matplotlib.pyplot as plt
import pickle

def Cov(x):
    m = len(x)
    Kmm = np.eye(m)
    for ii in range(m):
        for jj in range(ii+1,m):
            Kmm[ii,jj] = np.exp(-(x[ii]-x[jj])**2 )
            Kmm[jj,ii] = Kmm[ii,jj]

    return Kmm

def Cov2(x1,x2):
    m = len(x2)
    n = len(x1)
    Knm = np.zeros([n,m])
    for ii in range(n):
        for jj in range(m):
            Knm[ii, jj] = np.exp(-(x1[ii] - x2[jj]) ** 2 )
    return Knm

def solve(x,y, selection=True):
    n = len(x)

    # On peut soit sélectionner les points parmi les données disponibles :
    if selection:
        sel = [i for i in range(n)]
        ind = np.random.choice(sel, int(np.sqrt(n)), replace=False)
        x2 = [x[i] for i in ind]

    # Ou les prendre uniformément distribués
    else:
        x2 = np.linspace(-1, 1, 10)
        ind = []

    M = Cov2(x, x2)
    A = (0.5**2)*Cov(x2) + M.T @ M
    b = M.T @ y

    # Ici, le paramètre de régularisation nu vaut 1.0
    A = A + 1.*np.eye(int(np.sqrt(n)))

    # Il est utile de calculer les valeurs propres min/max de A,
    # mais seulement pour des petites matrices
    if n<101:
        ei, EI =np.linalg.eig(A)
        vv = [min(ei), max(ei)]
        print('Min and max eigenvalues of A : ', print(vv))

    alpha = np.linalg.solve(A,b)

    return alpha, ind

def plot_me(x,y, alpha, ind, selection=True):

    plt.plot(x,y,'o')

    xo = np.linspace(-1,1,100)
    if selection:
        x2 = [x[i] for i in ind]
    else:
        x2 = np.linspace(-1, 1, 10)


    yo = Cov2(xo, x2) @ alpha
    plt.plot(xo, yo, '-')
    plt.xlabel(r'$x$ feature')
    plt.ylabel(r'$y$ label')
    plt.grid()

    plt.show()


# Programme principal : exemple d'appel volontairement omis ici.
