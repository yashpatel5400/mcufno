# Painful crude verification that 1D winograd from `wincnn.py` matches with `np.convolve`

import wincnn
import numpy as np
import sympy as sp

# Numpy
print("The output of numpy convolution is: ")
print(np.convolve([1, 2], [3, 4, 5]))

# n is input size
# r is filter size

n = 2
r = 3
alpha = n + r - 1

# Winograd
AT, G, BT, f = wincnn.cookToomFilter((0, 1, -1), 2, 3, wincnn.FractionsInG)

B = BT.transpose()
A = AT.transpose()

di = wincnn.IndexedBase('d')
gi = wincnn.IndexedBase('g')

d = wincnn.Matrix(n, 1, lambda i, j: di[i])
g = wincnn.Matrix(r, 1, lambda i, j: gi[i])

V = A * d
U = G * g

M = U.multiply_elementwise(V)
Y = wincnn.simplify(B * M)


d0, d1, g0, g1, g2 = sp.symbols('d0 d1 g0 g1 g2')

Y_indexed = sp.Matrix([
    [            d0*g0],
    [d0*g1 + d1*g0],
    [d0*g2 + d1*g1],
    [            d1*g2]])

d_values = np.array([1, 2]) 
g_values = np.array([3, 4, 5])

subs = {d0: d_values[0], d1: d_values[1], g0: g_values[0], g1: g_values[1], g2: g_values[2]}

Y_num = Y_indexed.subs(subs)
Y_num = np.array(Y_num).astype(np.float64)

print("Winograd 1D Conv is: ")
print(Y_num)
