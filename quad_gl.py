# quad_gl.py
# Gauss–Legendre quadrature (n=2..7) + integration on R^2 segments.
import numpy as np

# --- Hard-coded Gauss–Legendre rules on [-1,1] for n=2..7 ---
# (points are symmetric; weights positive; exact for polynomials up to degree 2n-1)
_GL_TABLE = {
    2: (np.array([-1/np.sqrt(3), 1/np.sqrt(3)]),
        np.array([1.0, 1.0])),
    3: (np.array([ -np.sqrt(3/5), 0.0, np.sqrt(3/5)]),
        np.array([5/9, 8/9, 5/9])),
    4: (np.array([ -0.861136311594053, -0.339981043584856,
                    0.339981043584856,  0.861136311594053]),
        np.array([ 0.347854845137454,  0.652145154862546,
                   0.652145154862546,  0.347854845137454])),
    5: (np.array([ -0.906179845938664, -0.538469310105683,  0.0,
                    0.538469310105683,  0.906179845938664]),
        np.array([ 0.236926885056189,  0.478628670499366,  0.568888888888889,
                   0.478628670499366,  0.236926885056189])),
    6: (np.array([ -0.932469514203152, -0.661209386466265, -0.238619186083197,
                    0.238619186083197,  0.661209386466265,  0.932469514203152]),
        np.array([ 0.171324492379170,  0.360761573048139,  0.467913934572691,
                   0.467913934572691,  0.360761573048139,  0.171324492379170])),
    7: (np.array([ -0.949107912342759, -0.741531185599394, -0.405845151377397,
                   0.0,
                    0.405845151377397,  0.741531185599394,  0.949107912342759]),
        np.array([ 0.129484966168870,  0.279705391489277,  0.381830050505119,
                   0.417959183673469,
                   0.381830050505119,  0.279705391489277,  0.129484966168870])),
}

def gauss_legendre_rule(n: int):
    """
    Return (x, w) nodes and weights for Gauss–Legendre on [-1,1], for n=2..7.
    """
    if n not in _GL_TABLE:
        raise ValueError("Only n=2..7 supported by the TP; got n=%d" % n)
    x, w = _GL_TABLE[n]
    return x.copy(), w.copy()

def quad_1d(f, a: float, b: float, n: int):
    """
    Integrate f on [a, b] with n-point Gauss–Legendre (n=2..7).
    f: callable accepting vector x and returning vector values.
    """
    x, w = gauss_legendre_rule(n)
    xm = 0.5*(a + b)
    xr = 0.5*(b - a)
    xp = xm + xr * x  # map nodes
    return xr * np.sum(w * f(xp))

def quad_segment(g, y0: np.ndarray, y1: np.ndarray, n: int):
    """
    Integrate g over a straight segment [y0,y1] ⊂ R^2 with n-point Gauss–Legendre.
    g: callable accepting array of shape (q,2) and returning shape (q,)
       → evaluates g at physical points y_j
    y0, y1: array-like shape (2,)
    Returns: float (or complex) approximating ∫_seg g(y) ds
    """
    y0 = np.asarray(y0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    t, w = gauss_legendre_rule(n)             # t ∈ [-1,1]
    # Mapping: y(t) = 0.5[(1-t) y0 + (1+t) y1], ds = |y1 - y0|/2 dt
    Y = 0.5*((1.0 - t)[:,None]*y0[None,:] + (1.0 + t)[:,None]*y1[None,:])  # (n,2)
    jac = 0.5 * np.linalg.norm(y1 - y0)       # ds = jac * dt
    vals = g(Y)                                # (n,)
    return jac * np.sum(w * vals)

# ---------- Minimal self-tests ----------
if __name__ == "__main__":
    # 1D sanity: GL(n) integrates polynomials up to degree 2n-1 exactly
    for n in range(2, 8):
        a, b = -0.7, 1.3
        # degree 2n-1 polynomial
        m = 2*n - 1
        f = lambda x: x**m
        true = (b**(m+1) - a**(m+1)) / (m+1)
        approx = quad_1d(f, a, b, n)
        err = abs(approx - true)
        print(f"n={n}: poly deg {m}, abs error = {err:.3e}")

    # Segment sanity: integrate constant and linear functions
    y0 = np.array([0.0, 0.0])
    y1 = np.array([2.0, 1.0])
    L = np.linalg.norm(y1 - y0)

    const_g = lambda Y: np.ones(Y.shape[0])
    lin_g   = lambda Y: Y[:,0] + 2*Y[:,1]  # g(y)=x+2y

    for n in range(2, 8):
        I_const = quad_segment(const_g, y0, y1, n)
        # exact: ∫_seg 1 ds = length
        err_c = abs(I_const - L)

        # exact for linear g: average over endpoints × length
        g0 = lin_g(y0[None,:])[0]
        g1 = lin_g(y1[None,:])[0]
        I_lin_true = 0.5*(g0 + g1) * L
        I_lin = quad_segment(lin_g, y0, y1, n)
        err_l = abs(I_lin - I_lin_true)

        print(f"n={n}: segment const err={err_c:.3e}, linear err={err_l:.3e}")

