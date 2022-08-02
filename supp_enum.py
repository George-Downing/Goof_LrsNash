import math
import numpy as np
from numpy import linalg

import arithmetics

# algo: WHAT should player_1 play? (what is best response)
class What(object):
    def __init__(self, eps=1e-6):
        self.eps = eps
        return
    def __call__(self, A, y):
        # A: 2D array
        # y: 2D column vector/1D vector
        u = (A @ y).T
        umax = u.max()
        flags = u - umax >= -self.eps
        I = []
        for i, flag in enumerate(flags):
            if flag:
                I.append(i)
        I = np.array(I)
        return I

what = What(eps=1e-6)

# algo: WHEN should player_1 play I? (specific y, or, never)
class When(object):
    def __init__(self):
        self.eps = 1e-6
        return
    
    # A: 2D array, I: 1D array
    def __call__(self, A, I):
        m, n = A.shape
        k = min(len(I), m, n)
        Y = np.zeros([0, n])
        
        J = np.arange(0, k)
        while True:
            A_IJ = np.matrix(A[I[:, None], J])
            A_IJ_pinv = linalg.pinv(A_IJ)
            y_J = A_IJ_pinv * np.ones([k, 1])
            y_J /= y_J.sum(axis=0)
            y = np.zeros(n)
            y[J] = y_J.T[0]
            
            # check: make sure unselected strategies of A are sub-optimal
            y_min = y.min()
            if y_min > -self.eps:
                u_I_max = (A[I] @ y).max()
                u_max = (A @ y).max()
                if u_max - u_I_max <= self.eps:
                    Y = np.concatenate([Y, y[None, :]], axis=0)
            arithmetics.comb_incr(n, J)
            if arithmetics.comb_incr.overflow:
                break
        
        return Y

when = When()

def supp(x):
    return np.arange(x.shape[0])[x > 0]

def supp_enum(A, B):
    m, n = A.shape
    NashX = np.zeros([0, m])
    NashY = np.zeros([0, n])
    
    for k in range(1, 1 + min(m, n)):
        xs_table = np.empty([math.comb(n, k)], dtype=np.ndarray)
        ys_table = np.empty([math.comb(m, k)], dtype=np.ndarray)
        Is_table = np.empty([math.comb(n, k)], dtype=np.ndarray)
        Js_table = np.empty([math.comb(m, k)], dtype=np.ndarray)
        
        I = np.arange(k) # array of action
        while True:
            NoI = arithmetics.comb_enum(m, I)
            ys = when(A, I) # array of distribution
            if min(ys.shape):
                Js = np.apply_along_axis(supp, len(ys.shape)-1, ys) # array of support
            else:
                Js = np.empty([0, k], dtype=object)
            NoJs = np.array([arithmetics.comb_enum(n, J) for J in Js])
            # print("I:", I, "#", NoI, "when y in:", np.round(10000*ys).astype(int).tolist(), "Js:", Js.tolist(), "#", NoJs)
            ys_table[NoI] = ys
            Js_table[NoI] = NoJs
            arithmetics.comb_incr(m, I)
            if arithmetics.comb_incr.overflow:
                break

        J = np.arange(k)
        while True:
            NoJ = arithmetics.comb_enum(n, J)
            xs = when(B.T, J)
            if min(xs.shape):
                Is = np.apply_along_axis(supp, len(xs.shape)-1, xs)
            else:
                Is = np.empty([0, k], dtype=object)
            NoIs = np.array([arithmetics.comb_enum(m, I) for I in Is])
            # print("J:", J, "#", NoJ, "when x in:", np.round(10000*xs).astype(int).tolist(), "Is:", Is.tolist(), "#", NoIs)
            xs_table[NoJ] = xs
            Is_table[NoJ] = NoIs
            arithmetics.comb_incr(n, J)
            if arithmetics.comb_incr.overflow:
                break
        
        I = np.arange(k)
        while True:
            NoI = arithmetics.comb_enum(m, I)
            NoJs = Js_table[NoI]
            for j in range(NoJs.shape[0]):
                NoIs = Is_table[NoJs[j]]
                for i in range(NoIs.shape[0]):
                    if NoI == NoIs[i]:
                        NashX = np.concatenate([NashX, xs_table[NoJs[j]][i][None, :]], axis=0)
                        NashY = np.concatenate([NashY, ys_table[NoIs[i]][j][None, :]], axis=0)
            arithmetics.comb_incr(m, I)
            if arithmetics.comb_incr.overflow:
                break
    
    return NashX, NashY


def demo():
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    print("A:\n", A)
    print("B:\n", B)
    Amm = A.min(axis=1).min(axis=0)
    Bmm = B.min(axis=1).min(axis=0)

    NashX, NashY = supp_enum(A + Amm, B + Bmm)
    for i in range(NashX.shape[0]):
        x = NashX[i]
        y = NashY[i]
        u = x @ A @ y
        v = x @ B @ y
        print("#---=---=---=---" * 4)
        print("i:", i)
        print("x:", x)
        print("y:", y)
        print("u:", u)
        print("v:", v)
    return


if __name__ == "__main__":
    demo()
    