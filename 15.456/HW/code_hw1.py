import numpy as np
import scipy as sp
from fractions import Fraction
from math import gcd
from functools import reduce

# Question 1

A = np.array([[2, 3, 1], [1, 2, 0], [1/2, 1, 0]])
B = np.array([[3, 2, 1, 4, 2], [3, 3, 3, 4, 3], [0, 1, 2, 1, 0]])
C = np.array([[1, 16], [3, 1], [7, 4]])


def from_primitive_to_integers(nullspace, max_den=10):
    basis = []
    for j in range(nullspace.shape[1]):
        integers = np.array([int(f*np.lcm.reduce([f.denominator for f in [Fraction(val).limit_denominator(max_den)
                            for val in nullspace[:, j]]])) for f in [Fraction(val).limit_denominator(max_den) for val in nullspace[:, j]]])
        g = reduce(gcd, np.abs(integers))
        if g > 0:
            integers //= g
        basis.append(integers)
    return basis


rankA = np.linalg.matrix_rank(A)
nullspaceA = [from_primitive_to_integers(sp.linalg.null_space(A))]

rankB = np.linalg.matrix_rank(B)
nullspaceB = [from_primitive_to_integers(sp.linalg.null_space(B))]

rankC = np.linalg.matrix_rank(C)
nullspaceC = [from_primitive_to_integers(sp.linalg.null_space(C))]

print('rank(A) = ' + str(rankA))
print('ker(A) = ' + str(nullspaceA))

print('rank(B) = ' + str(rankB))
print('ker(B) = ' + str(nullspaceB))

print('rank(C) = ' + str(rankC))
print('ker(C) = ' + str(nullspaceC))


# Question 3

D = np.array([[2, 1, 1], [1, 1, 0], [0, 1, -1]])


rankD = np.linalg.matrix_rank(D)
nullspaceD = [from_primitive_to_integers(sp.linalg.null_space(D))]
print('rank(D) = ' + str(rankD))
print('ker(D) = ' + str(nullspaceD))

E = np.array([[2, 1, 0, 3, 1], [1, 1, 1, 2, 1], [0, 1, 2, 1, 0]])

rankE = np.linalg.matrix_rank(E)
nullspaceE = [from_primitive_to_integers(sp.linalg.null_space(E))]
print('rank(E) = ' + str(rankE))
print('ker(E) = ' + str(nullspaceE))


F = np.array([[2, 0], [1, 1], [0, 2]])
rankF = np.linalg.matrix_rank(F)
nullspaceF = [from_primitive_to_integers(sp.linalg.null_space(F))]


S = np.array([[1], [5]])

phi = np.dot(F, F.T)

print(phi)

print('rank(F) = ' + str(rankF))
print('ker(F) = ' + str(nullspaceF))
