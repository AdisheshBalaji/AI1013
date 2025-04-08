import math
import numpy as np



def unif(a, b):
    u = np.random.uniform(0, 1)
    return a + (b - a)*u

def bernoulli(p):
    u = np.random.uniform(0, 1)
    if u <= p:
        return 1
    else:
        return 0


def geometric(p):
    u = np.random.uniform(0, 1)
    return np.ceil(math.log(1 - u)/math.log(1 - p))
    


def exponential(t):
    u = np.random.uniform(0, 1)
    return -math.log(1 - u)/t


def rayleigh():
    u = np.random.uniform(0, 1)
    return math.sqrt(-2 * math.log(1 - u))


def gaussian(mu, var):
    u_1 = np.random.uniform([0, 1])
    u_2 = np.random.uniform([0, 1])

    R = rayleigh()
    theta = unif(0, 2 * math.pi)

    x_1 = R * math.cos(theta)
    x_2 = R * math.sin(theta)

    return math.sqrt(var) * x_1 + mu 









