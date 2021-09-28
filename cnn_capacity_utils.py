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

def compute_overlap(u, v):
    return u @ v / ((u@u)**(.5) * (v@v)**.5)

def compute_avg_overlap_img(u, v):
    if not type(u) == np.ndarray:
        u = np.array(u)
    if not type(v) == np.ndarray:
        v = np.array(v)
    u = u.reshape(*u.shape[:-2], -1)
    v = v.reshape(*v.shape[:-2], -1)
    udotv = np.sum(u * v, axis=-1)
    udotu = np.sum(u * u, axis=-1)
    vdotv = np.sum(v * v, axis=-1)
    overlap = udotv / (udotu**.5 * vdotv**.5)
    avg_overlap = np.mean(overlap)
    return avg_overlap
