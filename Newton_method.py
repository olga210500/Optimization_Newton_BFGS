import numpy as np
import math
from Half_divide_method import half_divide

np.seterr(divide='ignore', invalid='ignore')
def func(x,y):

    return math.log(1 + 3 * x ** 2 + 5 * y ** 2 + math.cos(x - y))

def funk_phi(a):
    hk = np.linalg.solve(matrixHesse(x, y), grad(x, y))
    x_a = x - hk[0] * a
    y_a = y - hk[1] * a

    return func(x_a, y_a)

def grad(x, y):

    diffX = (6 * x - math.sin(x - y)) / (1 + 3 * x ** 2 + 5 * y ** 2 + math.cos(x - y))
    diffY = (10 * y + math.sin(x - y)) / (1 + 3 * x ** 2 + 5 * y ** 2 + math.cos(x - y))
    gradient=np.array([[diffX, diffY]])
    return gradient.transpose()

def matrixHesse(x,y):

    x1=(6 - math.cos(x - y))/(3*x**2 + math.cos(x - y) + 5*y**2 + 1) - (6*x - math.sin(x - y))**2/(3*x**2 + math.cos(x - y) + 5*y**2 + 1)**2
    y1=math.cos(x - y)/(3*x**2 + math.cos(x - y) + 5*y**2 + 1) - ((6*x - math.sin(x - y))*(math.sin(x - y) + 10*y))/(3*x**2 + math.cos(x - y) + 5*y**2 + 1)**2
    x2=math.cos(x - y)/(3*x**2 + math.cos(x - y) + 5*y**2 + 1) - ((6*x - math.sin(x - y))*(math.sin(x - y) + 10*y))/(3*x**2 + math.cos(x - y) + 5*y**2 + 1)**2
    y2=(10 - math.cos(x - y))/(3*x**2 + math.cos(x - y) + 5*y**2 + 1) - (math.sin(x - y) + 10*y)**2/(3*x**2 + math.cos(x - y) + 5*y**2 + 1)**2
    matrix = np.array([[x1,y1],[x2,y2]])
    return matrix

x=0.001
y=0.0
eps=10**(-5)
k=0
xk = np.array([x, y])

print("Results")

while True:
    print("Iteration", k)
    print("Point x", xk)
    x = xk[0]
    y = xk[1]
    gradient = grad(x, y)
    norma = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    if norma <= eps:
        print(xk)
        break
    if norma > eps:
        hk=np.linalg.solve(matrixHesse(x,y), grad(x, y))
        ak=1
        xk[0]=x-ak*hk[0]
        xk[1]=y-ak*hk[1]
    if func(xk[0],xk[1])<func(x,y):
        if not np.sqrt((xk[0] - x) ** 2 + (xk[1] - y) ** 2) <= eps:
            print(xk)
            break

    else:
        ak = half_divide(funk_phi, [0.1, 1.0], eps)
        xk[0] = x - ak * hk[0]
        xk[1] = y - ak * hk[1]
    k = k + 1
