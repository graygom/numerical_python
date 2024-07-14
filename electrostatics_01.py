#
# TITLE: electrostatics 01
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: http://ael.cbnu.ac.kr/lectures/graduate/adv-em-1/2018-1/lecture-notes/electrostatics-1/ch4-electrostatics-1-appendix-C-numerical-sol-poisson.pdf
#
#

import time as tm
import numpy as np
from numba import jit
import scipy.sparse as scs
import matplotlib.pyplot as plt


#===============================================================================
# POINT CHARGE ~ 10min
# > iterative solution of 2D PDE Poisson's equation, electrostatics
#===============================================================================

if False:

    # geometry
    L = 1.0
    N = 128
    ds = L / N

    # 2D space
    x = np.linspace(0, L, N)
    y = np.copy(x)
    X, Y = np.meshgrid(x, y)

    # 2D charge density
    rho0 = 1.0
    rho = np.zeros((N, N))
    rho[int(round(N/2.0)), int(round(N/2.0))] = rho0

    # 2D electric potential
    V = np.zeros((N, N))

    # solving Poisson's equation using iterations
    iterations = 0
    eps = 1e-8              # convergence threshold
    error = 1e4             # current status

    while iterations < 1e5 and error > eps:
        # previous V
        Vprev = np.copy(V)
        error = 0.0

        # updating V
        for j in range(1, N-1):
            for i in range(1, N-1):
                V[i, j] = 0.25 * ( Vprev[i+1, j] + Vprev[i-1, j] + Vprev[i, j+1] + Vprev[i, j-1] + rho[i, j] * ds**2 )
                error += np.abs(V[i, j] - Vprev[i, j])

        iterations += 1
        error /= float(N)

    print('iterations = %i' % iterations)

    # plot
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    CS = plt.contour(X, Y, V, 30)               # contour plot

    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('2D electric potential of a point charge')

    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.show()



#===============================================================================
# POINT CHARGE ~ 6sec
# > iterative solution of 2D PDE Poisson's equation, electrostatics
#===============================================================================

if False:
    # geometry
    L = 1.0
    N = 256
    ds = L / N

    # 2D space
    x = np.linspace(0, L, N)
    y = np.copy(x)
    X, Y = np.meshgrid(x, y)

    # 2D charge density
    rho0 = 1.0
    rho = np.zeros((N, N))
    for j in range(int(round(N/2.0))-int(round(N/20.0)), int(round(N/2.0))+int(round(N/20.0))):
        rho[int(round(N/4.0*1.0)), j] = rho0
        rho[int(round(N/4.0*3.0)), j] = rho0

    # 2D electric potential
    V = np.zeros((N, N))
    Vprev = np.zeros((N, N))

    # solving Poisson's equation using iterations
    iterations = 0
    eps = 4e-7              # convergence threshold
    error = 1e4             # current status

    # numba
    @jit(nopython=True)
    def poisson(V, Vprev, rho, iterations, eps, error):
        while iterations < 1e6 and error > eps:
            # previous V
            Vprev = np.copy(V)
            error = 0.0

            # updating V
            for j in range(1, N-1):
                for i in range(1, N-1):
                    V[i, j] = 0.25 * ( Vprev[i+1, j] + Vprev[i-1, j] + Vprev[i, j+1] + Vprev[i, j-1] + rho[i, j] * ds**2 )
                    error += np.abs(V[i, j] - Vprev[i, j])

            iterations += 1
            error /= float(N)

        # return
        return V, iterations

    # numba
    V, iterations = poisson(V, Vprev, rho, iterations, eps, error)
    print('iterations = %i' % iterations)

    # plot
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    CS = plt.contour(X, Y, V, 30)               # contour plot

    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('2D electric potential of a point charge')

    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.show()



#===============================================================================
# Direct or Sparse Matrix Inversion
# > solution of 2D PDE Poisson's equation, electrostatics
#===============================================================================

if True:
    # time t0
    t0 = tm.time()

    # constants
    ep0 = 8.8541878176e-12    # F/m

    # geometry
    Nx = 512                  # column
    Ny = 128                   # row
    Lx = Nx * 1.0e-9          # meter
    Ly = Ny * 1.0e-9          # meter
    dLx = Lx / (Nx-1)         # meter
    dLy = Ly / (Ny-1)         # meter

    # vector plot
    x = np.linspace(0.0, Lx, Nx-1)
    y = np.linspace(0.0, Ly, Ny-1)
    X, Y = np.meshgrid(x, y)

    # matrix size
    N = Nx * Ny

    # matrix A, vector V, vector b
    A = np.zeros((N, N))
    V = np.zeros((N,1))
    b = np.zeros((N,1))

    # Boundary Conditions, dictionary
    BC = {}
    # Dirichlet BC, bottom
    for j in [0]:
        for i in range(Nx):
            BC['%i,%i' % (j, i)] = ['D', 0.0]
    # Dirichlet BC, top
    for j in [Ny-1]:
        for i in range(int(Nx/4.0*1.0), int(Nx/4.0*3.0)):
            BC['%i,%i' % (j, i)] = ['D', 1.0]
    # Neumann BC, top
    for j in [Ny-1]:
        for i in range(int(Nx/4.0*1.0)):
            BC['%i,%i' % (j, i)] = ['N', 0.0]
    # Neumann BC, top
    for j in [Ny-1]:
        for i in range(int(Nx/4.0*3.0), Nx):
            BC['%i,%i' % (j, i)] = ['N', 0.0]
    # Neumann BC, left
    for j in range(Ny):
        for i in [0]:
            BC['%i,%i' % (j, i)] = ['N', 0.0]
    # Nuemann BC, right
    for j in range(Ny):
        for i in [Nx-1]:
            BC['%i,%i' % (j, i)] = ['N', 0.0]

    # Direct Matrix Inversion
    for j in range(Ny):
        for i in range(Nx):
            node_id = '%i,%i' % (j, i)

            # SKIP at FDM
            if (j == 0 and i == 0) or (j == 0 and i == Nx-1) or (j == Ny-1 and i == 0) or (j == Ny-1 and i == Nx-1):
                A[Nx*j + i, Nx*j + i] = 1.0
                b[Nx*j + i, 0] = 0.0

            else:
                # boundary conditions
                if node_id in BC.keys():
                    if BC[node_id][0] == 'D':
                        A[Nx*j + i, Nx*j + i] = 1.0
                        b[Nx*j + i, 0] = BC[node_id][1]
                    elif BC[node_id][0] == 'N':
                        if i == 0:
                            A[Nx*j + i, Nx*j + i] = 1.0
                            A[Nx*j + i, Nx*j + i + 1] = -1.0
                            b[Nx*j + i, 0] = BC[node_id][1]
                        elif i == Nx-1:
                            A[Nx*j + i, Nx*j + i] = 1.0
                            A[Nx*j + i, Nx*j + i - 1] = -1.0
                            b[Nx*j + i, 0] = BC[node_id][1]
                        elif j == 0:
                            A[Nx*j + i, Nx*j + i] = 1.0
                            A[Nx*j + i, Nx*(j + 1) + i] = -1.0
                            b[Nx*j + i, 0] = BC[node_id][1]
                        elif j == Ny-1:
                            A[Nx*j + i, Nx*j + i] = 1.0
                            A[Nx*j + i, Nx*(j - 1) + i] = -1.0
                            b[Nx*j + i, 0] = BC[node_id][1]

                # normal nodes
                else:
                    A[Nx*j + i, Nx*j + i] = -4.0
                    A[Nx*j + i, Nx*j + i + 1] = 1.0
                    A[Nx*j + i, Nx*j + i - 1] = 1.0
                    A[Nx*j + i, Nx*(j + 1) + i] = 1.0
                    A[Nx*j + i, Nx*(j - 1) + i] = 1.0
                    b[Nx*j + i, 0] = 0.0

    # time t1
    t1 = tm.time()

    # diect matrix solver
    if False:
        V = np.linalg.solve(A, b)
        V = V.reshape((Ny, Nx))

    # time t2
    t2 = tm.time()

    # sparse matrix solver
    if True:
        As = scs.csr_matrix(A)
        AsLU = scs.linalg.splu(As)
        Vs = AsLU.solve(b)
        Vs = Vs.reshape((Ny, Nx))

    # electric field
    Ex = ( Vs[:, 1:] - Vs[:,:Nx-1] ) / dLx
    Ey = ( Vs[1:, :] - Vs[:Ny-1,:] ) / dLy
    Ex = (Ex[:Ny-1, :] + Ex[1:, :] ) / 2.0
    Ey = (Ey[:, :Nx-1] + Ey[:, 1:] ) / 2.0

    # time t3
    t3 = tm.time()

    # information
    print('initial setting time t1-t0 = %.6f sec' % (t1-t0))
    print('direct solver time t2-t1 = %.6f sec' % (t2-t1))
    print('sparse solver time t3-t2 = %.6f sec' % (t3-t2))

    # plot
    if True:
      fig, ax = plt.subplots(1, 4, figsize=(8,8))
      if False:
          ax[0].imshow(A)
          ax[0].grid(ls=':')
          ax[1].imshow(b)
          ax[1].grid(ls=':')
      ax[0].imshow(Vs)
      ax[1].imshow(Ex)
      ax[2].imshow(Ey)
      ax[3].quiver(Y, X, Ey, Ex)
      plt.show()



