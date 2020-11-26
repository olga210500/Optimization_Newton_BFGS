import numpy as np
import math
from DSK_algorithm import find_line_segment
from Half_divide_method import half_divide

np.seterr(divide='ignore', invalid='ignore')


def f(x, y):
    return math.log(1+3*x**2+5*y**2+math.cos(x-y))

def gradient(x, y):
    diffX=(6*x - math.sin(x - y))/(1 + 3*x**2 + 5*y**2 + math.cos(x - y))
    diffY=(10*y + math.sin(x - y))/(1 + 3*x**2 + 5*y**2 + math.cos(x - y))
    grad = np.array([[diffX, diffY]])
    return grad.transpose()
def func_phi(a):
    vector = np.dot(gradient(x, y).transpose(), H0)
    x_a = x - vector[0][0] * a
    y_a = y - vector[0][1] * a
    return f(x_a, y_a)

# Input data

x = 0.001
y = 0.0
h = 2
a0 = 1
eps = 10**(-5)
H0 = np.array([[1, 0], [0, 1]])
xk = np.array([x, y])
Hk = H0
k = 0
print("Results")
while True:
    print("Iteration", k)
    print("Point x", xk)
    print("Matrix\n", Hk)
    x = xk[0]
    y = xk[1]
    H0 = Hk
    gradientX = gradient(x, y)
    gradientXk = gradient(xk[0], xk[1])
    norma = np.sqrt(gradientXk[0] ** 2 + gradientXk[1] ** 2)
    if norma <= eps:
        print(xk)
        break
    if norma > eps:
        min = half_divide(func_phi, find_line_segment(func_phi, a0, h), eps)
        tempVector = -1*np.dot(gradientXk.transpose(), H0)
        xk[0] = x + tempVector[0][0] * min
        xk[1] = y + tempVector[0][1] * min
        gradientXk = gradient(xk[0], xk[1])
        yk = gradientXk - gradientX
        delta_xk = np.array([[xk[0] - x, xk[1] - y]])

        Hk_yk=np.dot(H0,yk)
        Hk_yk_yk=np.dot(Hk_yk.transpose(),yk)
        xk_yk = np.dot(delta_xk, yk)

        xk_xk_transope = np.dot(delta_xk.transpose(), delta_xk)
        xk_yk_transpose=np.dot(delta_xk.transpose(),yk.transpose())
        xk_yk_transpose_Hk=np.dot(xk_yk_transpose,H0)
        yk_xk_transpose=np.dot(yk,delta_xk)
        yk_xk_transpose_Hk=np.dot(H0,yk_xk_transpose)
        Hk=H0+(1.0+( Hk_yk_yk/ xk_yk))*(xk_xk_transope/xk_yk)-(xk_yk_transpose_Hk/xk_yk)-(yk_xk_transpose_Hk/xk_yk)
    k = k + 1
    if not np.sqrt((xk[0] - x) ** 2 + (xk[1] - y) ** 2) > eps:
        break
