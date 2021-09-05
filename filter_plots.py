import scipy.linalg
import numpy as np
from matplotlib import pyplot as plt

pi = np.pi
n = 19
a = .3
# Phi1 = scipy.linalg.dft(n)
# Ctilde = np.zeros((n,n))
# Ctilde[1,1] = a
# Ctilde[-1,-1] = a
# Phi1h = Phi1.conj().T
# Phi2 = np.array([[1, 1j],[1j, 1]])
# Phi2h = Phi2.conj().T
# C = np.zeros((2,n))
# C[0, 1] = a
# C[1,-1] = a


Phi1real = np.zeros((n,n))
# dx = n/(n-1)
# xx = np.arange(0, n, dx)
xx = np.arange(0, n)
# xx = np.linspace(0, n-1/n, n+1)
for i0, k in enumerate(range(0,n,2)):
    Phi1real[:, k] = np.cos(2*i0*pi*xx/n)
for i0, k in enumerate(range(1,n,2)):
    Phi1real[:, k] = np.sin(2*(i0+1)*pi*xx/n)
Phi1real = Phi1real / np.linalg.norm(Phi1real, axis=0)
# C = np.zeros((4,n))
# C[:4, 1:5] = a * np.eye(4,4)
C = np.zeros((2,n))
C[0, 1] = a
C[1, 2] = a
A = C @ Phi1real.T

Ctilde = np.zeros((n,n))
bvec = np.random.randn(n-2)
# bvec = np.zeros(n-2)
a = np.random.randn()
Ctilde[0,0] = bvec[0]
# Ctilde[0, 3:] = bvec[1:]
Ctilde[4:, 0] = bvec[2:]
Ctilde[1:3, 1:3] = a*np.eye(2,2)
Atilde = Ctilde @ Phi1real.T
# plt.imshow(Atilde); plt.show()

# Atilde2 = Phi1real @ Ctilde @ Phi1real.T
# plt.imshow(Atilde2); plt.show()
# Atilde = (Phi1 @ C1 @ Phi1h).real

def R(θ):
    return np.array([[np.cos(θ), -np.sin(θ)],
                     [np.sin(θ), np.cos(θ)]])
def rho2(g, n):
    return R(2 * pi * g / n)
def rho2tilde(g, n):
    rhom = np.zeros((n,n))
    rhom[1:3, 1:3] = R(2 * pi * g / n)
    rhom[0,0] = 1
    rhom[3:, 3:] = np.eye(n-3)
    return rhom 

def rho1(g, n):
    rhom = np.eye(n)
    rhom = np.roll(rhom, g, axis=1)
    return rhom

x = np.random.randn(n)
LHS = np.round(A @ rho1(1, n) @ x, 10)
RHS = np.round(rho2(1,n) @ A @ x, 10)

def B1(g, n):
    return Phi1real.T @ rho1(g, n) @ Phi1real

def B2(g, n):
    return rho2(g, n)
def B2tilde(g, n):
    b2 = np.zeros((n,n))
    b2[1:3, 1:3] = rho2(g,n)
    b2[0,0] = 1
    b2[3:,3:] = np.eye(n-3)
    return b2

LHS = Atilde @ rho1(1, n) @ x
RHS = rho2tilde(1,n) @ Atilde @ x

# np.round(B2tilde(1,n) @ Atilde, 10)
# np.round(Atilde @ rho1(1,n))

def rho2circ(g,n):
    return Phi1real @ B2tilde(g,n) @ Phi1real.T
Acirc = Phi1real @ Ctilde @ Phi1real.T
LHS = Acirc @ rho1(1, n) @ x
RHS = rho2circ(1,n) @ Acirc @ x
equiv_test = np.all(
    [np.allclose(Acirc @ rho1(k, n) @ x, rho2circ(k,n) @ Acirc @ x) for k in range(n)])

U, s , Vt = np.linalg.svd(Atilde)

# %% 
fig, ax = plt.subplots()
ax.imshow(A)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('first_mode_filters.pdf')
fig.show()
fig, ax = plt.subplots()
ax.imshow(Atilde)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('first_mode_filters_lifted.pdf')
fig.show()
fig, ax = plt.subplots()
ax.imshow(Acirc)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('first_mode_filters_circ.pdf')
fig.show()
# %% 


fig, ax = plt.subplots()
ax.imshow(Ar)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('first_mode_reduced_filters.pdf')
fig.show()
# fig, ax = plt.subplots()
# ax.imshow(Ar)
# fig.show()

def Crs(k, n):
    Crs = np.zeros((4,4))
    Crs[0, 0] = np.cos(2 * pi * k/n)
    Crs[0, 1] = -np.sin(2 * pi * k/n)
    Crs[1, 0] = np.sin(2 * pi * k/n)
    Crs[1, 1] = np.cos(2 * pi * k/n)
    Crs[2, 2] = np.cos(4 * pi * k/n)
    Crs[3, 2] = -np.sin(4 * pi * k/n)
    Crs[2, 3] = np.sin(4 * pi * k/n)
    Crs[3, 3] = np.cos(4 * pi * k/n)
    return Crs
evs = np.zeros((4,4), dtype=complex)
evs[:2,0] = [1, 1j] 
evs[:2,-1] = [1j, 1] 
evs[2:,1] = [1, 1j] 
evs[2:,2] = [1j, 1] 
evs = evs / np.linalg.norm(evs, axis=0)
# ew, ev1 = np.linalg.eig(Crs(1, n))
# ev1 = ev1 / np.linalg.norm(ev1, axis=0)
# ev1.conj().T @ Crs(1,n) @ ev1
# ew, ev2 = np.linalg.eig(Crs(2, n))
# ev2 = ev2 / np.linalg.norm(ev2, axis=0)
# ew4, ev4 = np.linalg.eig(Crs(4, n))
# ev = ev1 + ev2 + ev4
# ev = ev / np.linalg.norm(ev, axis=0)
# np.abs(evs.conj().T @ Crs(4,n) @ evs) > 1e-6
t = [np.all(np.diag(np.abs(evs.conj().T @ Crs(k,n) @ evs) > 1e-6)) for k in range(n)]
np.abs(evs.conj().T @ Crs(k,n) @ evs)
# evs.conj().T @ Crs(4,n) @ evs


# plt.imshow(Phireal.T @ Phireal); plt.show()
# ew, ev = np.linalg.eig(Crs(8, n))
rdummy = np.random.randn(n)
dummycirc = np.zeros((n,n))
for k in range(n):
    dummycirc[k] = np.roll(rdummy,k)
# ew, ev4 = np.linalg.eigh(dummycirc)
# rdummy = np.random.randn(n)
# dummycirc = np.zeros((n,n))
# for k in range(n):
    # dummycirc[k] = np.roll(rdummy,k)
# ew, evn = np.linalg.eigh(dummycirc)
# Crs0 = Crs(3, n)
# ew, ev = np.linalg.eig(Crs0)
Crs = np.zeros((4,n))
Crs[:4, 1:5] = a * np.eye(4,4)
C[2,2]=a
C[-2,-2]=a
# Cr[0,2] = a
# Cr[1,-2] = a
A = (Phi @ C @ Phis).real
Ar = Crs @ Phireal.T
evs @ Crs @ Phis

fig, ax = plt.subplots()
ax.imshow(A)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('first_and_second_mode_filters.pdf')
fig.show()

fig, ax = plt.subplots()
ax.imshow(Ar)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('first_and_second_mode_reduced_filters.pdf')
fig.show()

r1 = np.zeros(n)
r1[:5] = 1
Cs = np.zeros((n,n))
for k in range(n):
    Cs[k] = np.roll(r1,k-2)

fig, ax = plt.subplots()
ax.imshow(Cs)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('sparse_conv.pdf')
fig.show()

ew, ev = np.linalg.eigh(Cs)


C1 = 
