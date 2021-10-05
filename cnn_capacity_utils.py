import numpy as np
import matplotlib.pyplot as plt

def compute_pi_mean_reduced(L, k):
    base_v = np.zeros(L)
    base_v[::k] = 1 / (int(L/k))
    pi = np.zeros((k, L))
    for i in range(k):
        pi[i] = np.roll(base_v, i)
    return pi

def compute_pi_mean_reduced_2D(L, W, kx, ky):
    pi_mean_1d_L = compute_pi_mean_reduced(L, kx)
    pi_mean_1d_W = compute_pi_mean_reduced(W, ky)
    return np.kron(pi_mean_1d_L, pi_mean_1d_W)
