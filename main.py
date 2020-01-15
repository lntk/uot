import matplotlib.pyplot as plt
import numpy as np


def convergence():
    from uot import sinkhorn_uot, solve_g_dual_cp

    """
    HYPERPARAMETERS
    """
    range_a = 10
    range_b = 10
    range_C = 100
    dim_a = 100
    dim_b = 100
    eta = 0.05
    tau1 = 1.0
    tau2 = 1.0
    k = 100

    exp_name = f"[n={dim_a}]_[tau={'{0:.1f}'.format(np.mean(tau1))}]_[rC={range_C}]_[ra={range_a}]_[rb={range_b}]_[eta={'{0:.2f}'.format(np.mean(eta))}]"

    """
    INITIALIZATION
    """
    C = np.random.uniform(low=1, high=range_C, size=(dim_a, dim_b))
    C = (C + C.T) / 2
    a = np.random.uniform(low=0.1, high=range_a, size=(dim_a, 1))
    b = np.random.uniform(low=0.1, high=range_b, size=(dim_b, 1))

    """
    SOLVING UOT
    """
    g_dual_optimal, u_optimal, v_optimal = solve_g_dual_cp(C=C, a=a, b=b, eta=eta, tau=tau1)

    output = sinkhorn_uot(C=C, a=a, b=b, eta=eta, tau1=tau1, tau2=tau2, k=k)

    delta_u = [np.linalg.norm(u - u_optimal, ord=np.inf) for u in output["u"]]
    delta_v = [np.linalg.norm(v - v_optimal, ord=np.inf) for v in output["v"]]

    """
    PLOTTING
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    fig.suptitle(exp_name)

    axs[0, 0].plot(list(range(k)), output["f"], "r", label="f")
    axs[0, 0].plot([0, k - 1], [output["f_optimal"], output["f_optimal"]], "b", label="f optimal")
    axs[0, 0].set_title("f")
    axs[0, 0].legend()
    axs[0, 0].text(x=0.7, y=0.7, s=f"min_f={'{0:.2f}'.format(min(output['f']))}\noptimal_f={'{0:.2f}'.format(output['f_optimal'])}", horizontalalignment='center', verticalalignment='center', transform=axs[0, 0].transAxes)

    axs[0, 1].plot(list(range(k)), output["g_dual"], "r", label="g dual")
    axs[0, 1].plot([0, k - 1], [g_dual_optimal, g_dual_optimal], "b", label="g dual optimal")
    axs[0, 1].set_title("g dual")
    axs[0, 1].legend()
    axs[0, 1].text(x=0.7, y=0.7, s=f"min_g={'{0:.2f}'.format(min(output['g_dual']))}\noptimal_g={'{0:.2f}'.format(g_dual_optimal)}", horizontalalignment='center', verticalalignment='center', transform=axs[0, 1].transAxes)

    axs[1, 0].plot(list(range(k)), delta_u, "r", label="f optimal")
    axs[1, 0].set_title("||u - u*||_inf")
    axs[1, 0].text(x=0.7, y=0.7, s=f"min_du={'{0:.2f}'.format(delta_u[-1])}", horizontalalignment='center', verticalalignment='center', transform=axs[1, 0].transAxes)

    axs[1, 1].plot(list(range(k)), delta_v, "r", label="f optimal")
    axs[1, 1].set_title("||v - v*||_inf")
    axs[1, 1].text(x=0.7, y=0.7, s=f"min_dv={'{0:.2f}'.format(delta_v[-1])}", horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes)

    plt.show()


def rate():
    from uot import find_k_sinkhorn
    import time

    """
    HYPERPARAMETERS
    """
    range_a = 1
    range_b = 1
    range_C = 1
    dim_a = 100
    dim_b = 100
    tau1 = 1.0
    tau2 = 1.0

    """
    INITIALIZATION
    """
    C = np.random.uniform(low=0.01, high=range_C, size=(dim_a, dim_b))
    C = (C + C.T) / 2
    a = np.random.uniform(low=0.1, high=range_a, size=(dim_a, 1))
    b = np.random.uniform(low=0.1, high=range_b, size=(dim_b, 1))

    n = dim_a
    tau = tau1
    epsilon_list = np.linspace(start=10, stop=1, num=100, endpoint=False)
    num_eps = len(epsilon_list)

    alpha = np.sum(a)
    beta = np.sum(b)
    S = (alpha + beta + 1 / np.log(n)) * (2 * tau)
    T = 4 * ((alpha + beta) * (np.log(alpha + beta) + np.log(n)) + 1)

    print(alpha)
    print(beta)
    print(S)
    print(T)

    U = [S + T + epsilon + 2 * epsilon * np.log(n) / tau for epsilon in epsilon_list]
    eta_list = [epsilon_list[i] / U[i] for i in range(num_eps)]

    """
    SOLVING UOT
    """
    k_list = list()
    for i in range(num_eps):
        print(epsilon_list[i])
        print(eta_list[i])
        time.sleep(2)
        k_list.append(find_k_sinkhorn(C=C, a=a, b=b, epsilon=epsilon_list[i], eta=eta_list[i], tau1=tau1, tau2=tau2))

    """
    PLOTTING
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    axs[0, 0].plot(epsilon_list, k_list, "r", label="ratio")
    axs[0, 0].set_title("ratio")

    plt.show()


if __name__ == '__main__':
    rate()
