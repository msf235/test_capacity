import scipy.linalg
import numpy as np
from matplotlib import pyplot as plt

n = 20
Phi = scipy.linalg.dft(n)
C = np.zeros((n,n))
C[1,1]=.3
C[-1,-1]=.3
Phis = Phi.conj().T
A = (Phi @ C @ Phis).real
Phir = np.array([[],[]])
Phir = scipy.linalg.dft(2)
Phirs = Phir.conj().T
Cr = np.zeros((2,2))
Cr[0,0] = Cs[1,1]
Cr[1,1] = Cs[-1,-1]
Ar = (Phir @ Cr @ Phirs).real

fig, ax = plt.subplots()
ax.imshow(A)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('first_mode_filters.pdf')
fig.show()
# fig, ax = plt.subplots()
# ax.imshow(Ar)
# fig.show()

C[2,2]=.3
C[-2,-2]=.3
A = (Phi @ C @ Phis).real
fig, ax = plt.subplots()
ax.imshow(A)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.savefig('first_and_second_mode_filters.pdf')
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
