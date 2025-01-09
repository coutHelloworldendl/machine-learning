import numpy as np
import matplotlib.pyplot as plt

def sgn_mod(x):
    if x <= 0:
        return -1
    else:
        return 1

def decode_mod(H:np.array): #, x):
    n = H.shape[0]
    best_dist = 'inf'
    k = n - 1
    dist_k = 0
    E = np.zeros((n, n))
    u = np.around(E[k])
    u_best = np.zeros(n)
    y = (E[k][k] - u[k]) / H[k][k]
    step = np.zeros(n)
    step[k] = sgn_mod(y)
    U = set()
    while True:
        new_dist = dist_k + y**2
        if new_dist < (1 + 1e-10)*best_dist:
            if k != 0:
                for i in range(k-1):
                    E[k-1][i] = E[k][i] - y * H[k][i]
                k -= 1
                dist_k = new_dist
                u[k] = np.around(E[k][k])
                y = (E[k][k] - u[k]) / H[k][k]
                step[k] = sgn_mod(y)
            else:
                # u_best = u
                # best_dist = new_dist
                if new_dist != 0:
                    U.add(u)
                    best_dist = min(best_dist, new_dist)
                # k += 1
                u[k] += step[k]
                y = (E[k][k] - u[k]) / H[k][k]
                step[k] = -step[k] - sgn_mod(step[k])
        else:
            if k == n-1:
                return U
            else:
                k += 1
                u[k] += step[k]
                y = (E[k][k] - u[k]) / H[k][k]
                step[k] = -step[k] - sgn_mod(step[k])

def kissing_num(G:np.array): #, x):
    n = G.shape[0]
    while True:
        W = np.random.randint(low=-10, high=10, size=(n, n))
        det = np.linalg.det(W)
        if det == 1 or det == -1:
            break
    G2 = W @ G
    Q, G3 = np.linalg.qr(G2.T)
    G3 = G3.T
    Q = Q.T
    H3 = np.linalg.inv(G3)
    # x3 = x @ H3
    U3 = decode_mod(H3)
    best_num = 'inf'
    tau = 0
    for u in U3:
        best_num = min(best_num, np.linalg.norm(u @ G2, ord=2))
    for u in U3:
        if np.linalg.norm(u @ G2, ord=2) == best_num:
            tau += 1
    return tau


def theta_image(B:np.array):
    n = B.shape[0]
    A = B.T @ B
    u = np.zeros(n)
    eA = []
    for i in range(n):
        u = np.zeros(n)
        u[i] = 1
        M = u.T @ A @ u
        eA.append(M)
