import math
import numpy as np
from numpy.linalg  import inv, norm
from scipy.special import jv   # Функции Бесселя

import matplotlib.pyplot as plt

# Параметры задачи
#--------------------------------------
T  = 20
E  = 3.8
L  = 1
LL = L*L
def Potential(r, f):
    V0 = 0
    V1 = math.cos(T*f)
    V2 = 100*math.cos(T*f)/r/r
    V3 = 100*math.cos(T*f)**2/r/r
    return V1

# Параметры сетки
#--------------------------------------

# Область решения [0, F] по углу и [0, R] по радиусу
F = 2*math.pi/T
R = 15

# Количество узлов и шаг сетки
N  = 1 << 6
h  = R/N
hh = h*h

M  = 1 << 6
d  = F/M
dd = d*d

# Задание матриц
#--------------------------------------

I  = np.identity(N, dtype=np.float64)

# Сеточные значения потенциала

V  = np.zeros((M, N), dtype=np.float64)
m = 0
while m != M:
    f = m*d
    n = 0
    while n != N:
        r = h*(n + 1)
        V[m][n] = Potential(r, f)
        n += 1
    m += 1

# Блочные элементы +-1 диагоналей A

b  = (1/dd + 1j*L/d)*np.array([1/hh/n/n for n in range(1, N + 1)])
bh = b.conj()
BH = I*bh

# Блочные элементы главной(0) диагонали A

C  = np.zeros((N, N), dtype=np.float64)

r  = h
rr = r*r
rh = r*h
C[0][0] = -1/hh + 3/2/rh + 2/dd/rr + LL/rr
C[0][1] =  2/hh - 2/rh
C[0][2] = -1/hh + 1/2/rh

r  = 2*h
rr = r*r
rh = r*h
C[1][0] = -1/hh + 1/3/rh
C[1][1] =  2/hh + 1/2/rh + 2/dd/rr + LL/rr
C[1][2] = -1/hh - 1/rh
C[1][3] =  0/hh + 1/6/rh

n = 2
while n != N - 2:
    r  = h*(n + 1)
    rr = r*r
    rh = r*h
    C[n][n + 2] = 1/12/hh + 1/12/rh
    C[n][n + 1] = -4/3/hh - 2/3/rh
    C[n][n]     =  5/2/hh + 2/dd/rr + LL/rr
    C[n][n - 1] = -4/3/hh + 2/3/rh
    C[n][n - 2] = 1/12/hh - 1/12/rh
    n += 1

r  = h*(N - 1)
rh = r*h
rr = r*r
C[N - 2][N - 1] = -1/hh - 1/3/rh
C[N - 2][N - 2] =  2/hh - 1/2/rh + 2/dd/rr + LL/rr
C[N - 2][N - 3] = -1/hh + 1/rh
C[N - 2][N - 4] =  0/hh - 1/6/rh

r  = h*N
rr = r*r
rh = r*h
C[N - 1][N - 1] = -1/hh - 3/2/rh + 2/dd/rr + LL/rr
C[N - 1][N - 2] =  2/hh + 2/rh
C[N - 1][N - 3] = -1/hh - 1/2/rh

# Умножение столбца на блочную матрицу A
#--------------------------------------
def block_dot(U):
    y = np.zeros((N, M), dtype=np.complex128)

    m = 0
    while m != M:
        y[m] = -(C + 2*np.diag(V[m])).dot(U[m]) + b*U[(m + 1)&(M - 1)] + bh*U[(m - 1)&(M - 1)]
        m += 1
    
    return y

# Метод циклической прогонки для трехдиагональной блочной матрицы A
#--------------------------------------
def block_tridiagonal_solution(mu, f):
    
    # Прогоночные коэффициенты(матрицы)
    alpha = np.zeros((M + 1, N, N), dtype=np.complex128)
    beta  = np.zeros((M + 1, N),    dtype=np.complex128)
    gamma = np.zeros((M + 1, N, N), dtype=np.complex128)
    gamma[0] = np.identity(N, dtype=np.complex128)

    # Прямая прогонка
    m = 0
    while m != M:
        Cm = C + 2*np.diag(V[m]) + mu*I - BH.dot(alpha[m])
        Cm_inv = inv(Cm)
        alpha[m + 1] = Cm_inv*b
        beta[m + 1]  = Cm_inv.dot(f[m] + BH.dot(beta[m]))
        gamma[m + 1] = np.dot(Cm_inv*bh, gamma[m])
        m += 1
    
    # Обратная прогонка
    P = np.zeros(N, dtype=np.complex128)
    Q = np.identity(N, dtype=np.complex128)

    m = M - 2
    while m != -1:
        P = alpha[m + 1].dot(P) + beta[m + 1]
        Q = alpha[m + 1].dot(Q) + gamma[m + 1]
        m -= 1
    
    y = np.zeros((M, N), dtype=np.complex128)
    y[M - 1] = inv(I - alpha[M].dot(Q) - gamma[M]).dot(beta[M] + alpha[M].dot(P))

    m = M - 2
    while m != -1:
        y[m] = alpha[m + 1].dot(y[m + 1]) + beta[m + 1] + gamma[m + 1].dot(y[M - 1])
        m -= 1
    
    return y/norm(y)

# Метод итераций Рэлея
#--------------------------------------

# Начальное приближение в виде точного решения в отсутствии потенциала
U = np.ones((M, N), dtype=np.complex128)
U = U*np.array([jv(L, math.sqrt(2*E)*h*(n + 1)) for n in range(0, N)])

mu = -2*E # Начальное приближение к собственному значению

i = 0
while i != 10:

    U = block_tridiagonal_solution(mu, -U)
    mu_last = mu
    mu = (np.vdot(U, block_dot(U))/np.vdot(U, U)).real
    
    if math.fabs(mu - mu_last) < 0.0001:
        break
    i += 1

# Точное решение при V = 0
#--------------------------------------

K = N << 1
k = R/K
u = np.zeros(K, dtype=np.float64) # Сеточные значения точного решения

n = 0
while n != K:
    r = k*n
    u[n] = jv(L, math.sqrt(-mu)*r)
    n += 1

# Графики решений
#--------------------------------------

u = u/norm(u)*math.sqrt(K/N/M)
U = U/norm(U)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(12, 5)
fig.suptitle(r'$E = $' + f'{-mu/2:.2},' + r'$\quad L = $' + f'{L},' + r'$\quad T = 2\pi/$' + f'{T}')
plt.subplots_adjust(left = 0.08,
                    right= 0.97,
                    wspace= 0.05)

# Зависимость по радиусу
plt.subplot(1, 2, 1)

m = M >> 1
sgn = 1 - 2*(U[m][2].real < 0)

ax1.plot(np.arange(0, K)*k, u,
         linestyle = '-',
         #linewidth = 1,
         color = 'g',
         label=r'$V(\rho, \varphi) = 0$'
         )

ax1.plot(np.arange(1, N + 1)*h, sgn*U[m].real,
         marker = 'o',
         markersize = 4,
         linestyle = '-',
         linewidth = 0.5,
         color = 'k',
         label='Numerical'
         )

ax1.axis([0,R,
          math.floor(100*min(min(u), min(sgn*U[m].real)))/100 - 0.01,
          math.ceil(100*max(max(u), max(sgn*U[m].real)))/100 + 0.01,
          ])
ax1.set_xlabel(r'$\rho$')
ax1.set_ylabel(r'$\mathsf{Re}(u)$')
ax1.set_title(r'$\varphi$ = ' + f'{m*d:.2}')
ax1.grid(True)
ax1.legend(fontsize=14)

# Зависимость по полярному углу
n = N >> 1

ax2.set_xlim(0, F)

ax2.plot([0, F], [float(u[(n + 1) << 1])]*2,
         linestyle = '-',
         #linewidth = 1,
         color = 'g'
         )

ax2.plot(np.arange(0, M)*d, sgn*U.T[n].real,
         marker = 'o',
         markersize = 4,
         linestyle = '-',
         linewidth = 0.5,
         color = 'k'
         )

ax2.set_xlabel(r'$\varphi$')
ax2.set_title(r'$\rho = $' + f'{(n + 1)*h:.2}')
ax2.grid(True)


#fig.savefig('test.png', dpi=500)
plt.show()

