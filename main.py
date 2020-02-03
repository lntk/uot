import matplotlib.pyplot as plt
import numpy as np


def max_norm(x):
    return np.amax(np.abs(x))


def setting():
    np.random.seed(73)

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
    C = np.random.uniform(low=1, high=range_C, size=(dim_a, dim_b)).astype("float128")
    C = (C + C.T) / 2
    a = np.random.uniform(low=0.1, high=range_a, size=(dim_a, 1)).astype("float128")
    b = np.random.uniform(low=0.1, high=range_b, size=(dim_b, 1)).astype("float128")

    return dim_a, dim_b, eta, tau1, tau2, k, C, a, b, exp_name


def convergence():
    from uot import sinkhorn_uot, solve_g_dual_cp

    dim_a, dim_b, eta, tau1, tau2, k, C, a, b, exp_name = setting()

    """
    SOLVING UOT
    """
    g_dual_optimal, u_optimal, v_optimal = solve_g_dual_cp(C=C, a=a, b=b, eta=eta, tau=tau1)

    output = sinkhorn_uot(C=C, a=a, b=b, eta=eta, tau1=tau1, tau2=tau2, k=k)

    delta_u = [max_norm(u - u_optimal) for u in output["u"]]
    delta_v = [max_norm(v - v_optimal) for v in output["v"]]
    delta_k = [max(delta_u[i], delta_v[i]) for i in range(k)]

    # delta_u_ratio = [delta_u[i] / delta_u[i + 1] for i in range(k - 1)]
    # delta_v_ratio = [delta_v[i] / delta_v[i + 1] for i in range(k - 1)]
    delta_list = [delta_k[i] / delta_k[i + 1] for i in range(k - 1)]

    """
    PLOTTING
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

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


def theorems_and_lemmas():
    from uot import sinkhorn_uot, solve_g_dual_cp, compute_B

    dim_a, dim_b, eta, tau1, tau2, k, C, a, b, exp_name = setting()
    n = dim_a
    tau = tau1

    R = max(max_norm(np.log(a)), max_norm(np.log(b))) + max(np.log(n), 1 / eta * max_norm(C) - np.log(n))

    """
    SOLVING UOT
    """
    g_dual_optimal, u_optimal, v_optimal = solve_g_dual_cp(C=C, a=a, b=b, eta=eta, tau=tau1)
    output = sinkhorn_uot(C=C, a=a, b=b, eta=eta, tau1=tau1, tau2=tau2, k=k)

    """
    LEMMA 1
    """
    B_star = compute_B(C=C, u=u_optimal, v=v_optimal, eta=eta)
    a_star = np.sum(B_star, axis=1, keepdims=True)
    b_star = np.sum(B_star, axis=0, keepdims=True).T
    lhs = list(u_optimal / tau + 0.1)
    rhs = list(np.log(a) - np.log(a_star))

    delta_u = [max_norm(u - u_optimal) for u in output["u"]]
    delta_v = [max_norm(v - v_optimal) for v in output["v"]]
    delta_k_list = [max(delta_u[i], delta_v[i]) for i in range(k)]

    bound_list = [np.power(tau / (tau + eta), i) * tau * R for i in range(k)]

    """
    PLOTTING
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    fig.suptitle(exp_name)

    axs[0, 0].plot(list(range(n)), lhs, "r")
    axs[0, 0].plot(list(range(n)), rhs, "b")
    axs[0, 0].set_xlabel("k (-th iteration)")
    axs[0, 0].set_title("Lemma 1")

    plt.show()


def rate():
    from uot import find_k_sinkhorn, solve_f_cp
    from tqdm import tqdm

    """
    HYPERPARAMETERS
    """
    min_a = 0.01
    min_b = 0.01
    min_C = 0.01
    max_a = 5
    max_b = 5
    max_C = 100
    dim_a = 5
    dim_b = 5
    tau1 = 5
    tau2 = 5

    """
    INITIALIZATION
    """
    C = np.random.uniform(low=min_a, high=max_C, size=(dim_a, dim_b)).astype("float128")
    C = (C + C.T) / 2
    a = np.random.uniform(low=min_b, high=max_a, size=(dim_a, 1)).astype("float128")
    b = np.random.uniform(low=min_C, high=max_b, size=(dim_b, 1)).astype("float128")

    # a = a / np.sum(a) * 1
    # b = b / np.sum(b) * 2

    n = dim_a
    tau = tau1
    epsilon_list = np.linspace(start=1, stop=0.5, num=100, endpoint=False)
    epsilons = np.array(epsilon_list)
    num_eps = len(epsilon_list)

    # scale = 2
    # a = a * scale
    # b = b * scale
    # C = C * scale
    # tau = tau * scale
    # epsilons = epsilons * scale
    # epsilon_list = list(epsilons)

    alpha = np.sum(a)
    beta = np.sum(b)
    S = 1 / 2 * (alpha + beta) + 1 / 2 + 1 / (4 * np.log(n))
    T = 1 / 2 * (alpha + beta) * (np.log((alpha + beta) / 2) + 2 * np.log(n) - 1) + np.log(n) + 5 / 2

    C1 = np.e * tau * (S + T)
    C2 = np.log(6 * max_norm(C)) + np.log(tau * (tau + 1)) + np.log(S + T)

    U_list = [max(S + T, epsilon, 4 * epsilon * np.log(n) / tau, 4 * epsilon * (alpha + beta) * np.log(n) / tau) for epsilon in epsilon_list]
    eta_list = [epsilon_list[i] / U_list[i] for i in range(num_eps)]
    R_list = [max(max_norm(np.log(a)), max_norm(np.log(b))) + max(np.log(n), 1 / eta_list[i] * max_norm(C) - np.log(n)) for i in range(num_eps)]

    """
    SOLVING UOT
    """
    f_optimal, _ = solve_f_cp(C, a, b, tau1=tau1, tau2=tau2)

    k_list_empirical_true = list()
    for i in tqdm(range(num_eps)):
        k_list_empirical_true.append(find_k_sinkhorn(C=C, a=a, b=b, epsilon=epsilon_list[i], f_optimal=f_optimal, eta=eta_list[i], tau1=tau1, tau2=tau2, momentum=1000))

    k_list_empirical_first = list()
    for i in tqdm(range(num_eps)):
        k_list_empirical_first.append(find_k_sinkhorn(C=C, a=a, b=b, epsilon=epsilon_list[i], f_optimal=f_optimal, eta=eta_list[i], tau1=tau1, tau2=tau2, momentum=0))

    k_list_formula = [(tau * U_list[i] / epsilon_list[i] + 1) * (np.log(8 * eta_list[i] * R_list[i]) + np.log(tau * (tau + 1)) + 3 * np.log(U_list[i] / epsilon_list[i])) for i in range(num_eps)]
    # k_list_function = [C1 / epsilon * (C2 - np.log(epsilon)) for epsilon in epsilon_list]

    print("alpha: ", alpha)
    print("beta: ", beta)
    print("S: ", S)
    print("T: ", T)
    print("U: ", U_list)
    print("R: ", R_list)
    print("eta: ", eta_list)
    print("f_optimal: ", f_optimal)
    print("K empirical true: ", k_list_empirical_true)
    print("K empirical first: ", k_list_empirical_first)
    print("K formula: ", k_list_formula)

    """
    SOME FUNCTIONS OF EPSILON
    """
    # f_eps_1 = list(1 / epsilons)
    # f_eps_2 = list(1 / np.power(epsilons, 0.5))
    # f_eps_3 = list(1 / np.power(epsilons, 0.1))

    f_eps_1 = list(1 / epsilons * k_list_empirical_true[-1] * epsilons[-1])
    f_eps_2 = list(1 / np.power(epsilons, 0.5) * k_list_empirical_true[-1] * np.power(epsilons[-1], 0.5))
    f_eps_3 = list(1 / np.power(epsilons, 0.1) * k_list_empirical_true[-1] * np.power(epsilons[-1], 0.1))

    # k_formula = np.array(k_list_formula)
    # k_list_formula_scaled = list(k_formula / k_list_formula)

    """
    PLOTTING
    """

    plt.figure(figsize=(8, 6))
    plt.plot(epsilon_list, k_list_empirical_true, "r", label=r"$k_{true}$")
    plt.plot(epsilon_list, k_list_empirical_first, "g", label=r"$k_{first}$")
    plt.plot(epsilon_list, k_list_formula, "b", label=r"$k_{formula}$")
    # plt.plot(epsilon_list, k_list_function, "g", label="function")
    # plt.plot(epsilon_list, f_eps_1, "g", label="1/e")
    # plt.plot(epsilon_list, f_eps_2, "m", label="1/e^0.5")
    # plt.plot(epsilon_list, f_eps_3, "y", label="1/e^0.1")
    plt.xlabel("epsilon")
    plt.ylabel("k (iterations)")
    plt.legend()

    plt.show()


def test():
    B = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    print(B)
    print(B.shape)
    print(np.sum(B, axis=0, keepdims=True).shape)
    print(np.sum(B, axis=1, keepdims=True).shape)


if __name__ == '__main__':
    # convergence()
    # rate()
    theorems_and_lemmas()
    # test()
