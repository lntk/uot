import numpy as np
from numpy.linalg import norm
from copy import copy
import cvxpy as cp


def compute_entropy(P):
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)


def compute_KL(P, Q):
    log_ratio = np.log(P) - np.log(Q)
    return np.sum(P * log_ratio - P + Q)


def dot(x, y):
    return np.sum(x * y)


def compute_B(C, u, v, eta):
    return np.exp((u + v.T - C) / eta)


def compute_f(C, X, a, b, tau1, tau2):
    Xa = X.sum(axis=1).reshape(-1, 1)
    Xb = X.sum(axis=0).reshape(-1, 1)

    return dot(C, X) + tau1 * compute_KL(Xa, a) + tau2 * compute_KL(Xb, b)


def compute_g_primal(C, u, v, a, b, eta, tau1, tau2):
    B = compute_B(C, u, v, eta)

    Ba = B.sum(axis=1).reshape(-1, 1)
    Bb = B.sum(axis=0).reshape(-1, 1)

    return dot(C, B) + tau1 * compute_KL(Ba, a) + tau2 * compute_KL(Bb, b) - eta * compute_entropy(B)


def compute_g_dual(C, u, v, a, b, eta, tau1, tau2):
    B = compute_B(C, u, v, eta)
    f = eta * np.sum(B) + tau1 * dot(np.exp(- u / tau1), a) + tau2 * dot(np.exp(- v / tau2), b)

    return f


def solve_g_dual_cp(C, a, b, eta, tau):
    u = cp.Variable(shape=a.shape)
    v = cp.Variable(shape=b.shape)

    u_stack = cp.vstack([u.T for _ in range(100)])
    v_stack = cp.hstack([v for _ in range(100)])

    # obj = eta * cp.sum(cp.multiply(cp.exp(u + v.T) * cp.exp(v).T, 1 / cp.exp(C)))
    obj = eta * cp.sum(cp.multiply(cp.exp(u_stack + v_stack), 1 / cp.exp(C)))
    obj += tau * cp.sum(cp.multiply(cp.exp(- u / tau), a))
    obj += tau * cp.sum(cp.multiply(cp.exp(- v / tau), b))

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()

    return prob.value, u.value, v.value


def solve_f_primal_cp(C, a, b, tau1=1.0, tau2=1.0):
    """
    Convex programming solver for standard Unbalanced Optimal Transport.

    :param C:
    :param a:
    :param b:
    :param tau1:
    :param tau2:
    :return:
    """

    X = cp.Variable((a.shape[0], b.shape[0]), nonneg=True)

    row_sums = cp.sum(X, axis=1)
    col_sums = cp.sum(X, axis=0)

    obj = cp.sum(cp.multiply(X, C))

    obj -= tau1 * cp.sum(cp.entr(row_sums))
    obj -= tau2 * cp.sum(cp.entr(col_sums))

    obj -= tau1 * cp.sum(cp.multiply(row_sums, cp.log(a.reshape(-1, ))))
    obj -= tau2 * cp.sum(cp.multiply(col_sums, cp.log(b.reshape(-1, ))))

    obj -= (tau1 + tau2) * cp.sum(X)
    obj += tau1 * cp.sum(a.reshape(-1, )) + tau2 * cp.sum(b.reshape(-1, ))

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()

    return prob.value, X.value


def gd_uot(C, a, b, tau1=1.0, tau2=1.0):
    """
    Gradient descent solver for standard Unbalanced Optimal Transport.

    :param C:
    :param a:
    :param b:
    :param tau1:
    :param tau2:
    :return:
    """
    import torch

    Ct = torch.tensor(C, requires_grad=False)
    at = torch.tensor(a, requires_grad=False)
    bt = torch.tensor(b, requires_grad=False)

    pass


def sinkhorn_uot(C, a, b, eta=1.0, tau1=1.0, tau2=1.0, k=100):
    """
    Sinkhorn algorithm for entropic-regularized Unbalanced Optimal Transport.

    :param C:
    :param a:
    :param b:
    :param eta:
    :param tau1:
    :param tau2:
    :param k:
    :param epsilon:
    :return:
    """

    output = {
        "u": list(),
        "v": list(),
        "f": list(),
        "g_dual": list()
    }

    # Compute optimal value and X for unregularized UOT
    f_optimal, X_optimal = solve_f_primal_cp(C, a, b, tau1=tau1, tau2=tau2)
    output["f_optimal"] = f_optimal
    output["X_optimal"] = X_optimal

    # Initialization
    u = np.zeros_like(a)
    v = np.zeros_like(b)

    # # Compute initial value of f
    # B = compute_B(C, u, v, eta)
    # f = compute_f_primal(C=C, X=B, a=a, b=b, tau1=tau1, tau2=tau2)
    # output["f"].append(f)

    for i in range(k):
        u_old = copy(u)
        v_old = copy(v)
        B = compute_B(C, u, v, eta)

        f = compute_f(C=C, X=B, a=a, b=b, tau1=tau1, tau2=tau2)

        output["f"].append(f)

        # Sinkhorn update
        if i % 2 == 0:
            Ba = B.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(a) - np.log(Ba)) * (tau1 * eta / (eta + tau1))
        else:
            Bb = B.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(b) - np.log(Bb)) * (tau2 * eta / (eta + tau2))

        g_dual = compute_g_dual(C=C, u=u, v=v, a=a, b=b, eta=eta, tau1=tau1, tau2=tau2)

        output["u"].append(u)
        output["v"].append(v)
        output["g_dual"].append(g_dual)

        err = norm(u - u_old, ord=1) + norm(v - v_old, ord=1)

        # if err < 1e-10:
        #     break
        #
        # if np.abs(f - output["f_optimal"]) < epsilon:
        #     break

    return output


def find_k_sinkhorn(C, a, b, epsilon, eta=1.0, tau1=1.0, tau2=1.0):
    # Initialization
    u = np.zeros_like(a)
    v = np.zeros_like(b)

    i = 0

    while True:
        B = compute_B(C, u, v, eta)

        f_primal = compute_f(C=C, X=B, a=a, b=b, tau1=tau1, tau2=tau2)

        # Sinkhorn update
        if i % 2 == 0:
            Ba = B.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(a) - np.log(Ba)) * (tau1 * eta / (eta + tau1))
        else:
            Bb = B.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(b) - np.log(Bb)) * (tau2 * eta / (eta + tau2))

        if np.abs(f_primal - f_optimal) < epsilon:
            return i

        i += 1
