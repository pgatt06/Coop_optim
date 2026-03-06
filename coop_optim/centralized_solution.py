import numpy as np
import pickle

def _as_1d_float_array(x):
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)

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

def build_nystrom_matrices(x, y, n=100, m=10, selection=True, seed=0):
    """
    Construit les matrices noyau de Nyström pour des entrées 1D
    avec le noyau RBF k(x,z)=exp(-(x-z)^2).

    Renvoie :
        x_n: tableau (n,) des entrées
        y_n: tableau (n,) des labels
        x2:  tableau (m,) des points de référence (landmarks)
        M:   matrice (n,m) avec M[i,j] = k(x_n[i], x2[j])
        Kmm: matrice (m,m) avec Kmm[i,j] = k(x2[i], x2[j])
        ind: indices des landmarks dans x_n (vide si selection=False)
    """
    rng = np.random.default_rng(seed)
    x_n = _as_1d_float_array(x)[:n]
    y_n = _as_1d_float_array(y)[:n]

    if selection:
        ind = rng.choice(np.arange(len(x_n)), size=m, replace=False)
        x2 = x_n[ind]
    else:
        x2 = np.linspace(-1, 1, m)
        ind = np.array([], dtype=int)

    M = Cov2(x_n, x2)
    Kmm = Cov(x2)
    return x_n, y_n, _as_1d_float_array(x2), np.asarray(M, dtype=float), np.asarray(Kmm, dtype=float), ind

def solve_from_matrices(M, y, Kmm, sigma=0.5, nu=1.0):
    """
    Résout le problème centralisé de régression ridge noyau :
        min_a  0.5||M a - y||^2 + 0.5*sigma^2*a^T Kmm a + 0.5*nu||a||^2
    """
    M = np.asarray(M, dtype=float)
    y = _as_1d_float_array(y)
    Kmm = np.asarray(Kmm, dtype=float)
    m = Kmm.shape[0]

    A = (sigma**2) * Kmm + M.T @ M + nu * np.eye(m)
    b = M.T @ y
    alpha = np.linalg.solve(A, b)
    return alpha

def objective(alpha, M, y, Kmm, sigma=0.5, nu=1.0):
    alpha = _as_1d_float_array(alpha)
    y = _as_1d_float_array(y)
    M = np.asarray(M, dtype=float)
    Kmm = np.asarray(Kmm, dtype=float)
    res = M @ alpha - y
    return 0.5 * (res @ res) + 0.5 * (sigma**2) * (alpha @ (Kmm @ alpha)) + 0.5 * nu * (alpha @ alpha)

def gradient(alpha, M, y, Kmm, sigma=0.5, nu=1.0):
    alpha = _as_1d_float_array(alpha)
    y = _as_1d_float_array(y)
    M = np.asarray(M, dtype=float)
    Kmm = np.asarray(Kmm, dtype=float)
    return M.T @ (M @ alpha - y) + (sigma**2) * (Kmm @ alpha) + nu * alpha

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
    import matplotlib.pyplot as plt

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


"""
Programme principal
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    with open('first_database.pkl', 'rb') as f:
        x,y = pickle.load(f)

    num_points = 100
    alpha, ind = solve(x[:num_points],y[:num_points], selection=True)

    print('Result summary -----------------')
    print('Optimal centralised alpha = ', alpha)

    plot_me(x[:num_points],y[:num_points], alpha, ind, selection=True)
