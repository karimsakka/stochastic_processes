import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_gbm_simulation(S0=100.0, mu=0.05, sigma=0.2, T=1.0, N=252, num_paths=10000):
    """
    Simulates Geometric Brownian Motion (GBM) paths using Monte Carlo integration
    and compares the empirical expected price E[S_T] with the theoretical value S0 * exp(mu * T).
    """
    np.random.seed(42)
    dt = T / N
    t_values = np.linspace(0, T, N + 1)

    # Generate Brownian increments dB ~ N(0, dt) for all paths
    dB = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_paths, N))
    B = np.zeros((num_paths, N + 1))
    B[:, 1:] = np.cumsum(dB, axis=1)

    # Analytical solution for GBM: S(t) = S0 * exp((mu - 0.5 * sigma^2)*t + sigma * B_t)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t_values + sigma * B)

    # Empirical vs Theoretical Expected Terminal Value E[S_T]
    S_T_empirical = np.mean(S[:, -1])
    S_T_theoretical = S0 * np.exp(mu * T)

    print("--- Monte Carlo Simulation for SDEs (Geometric Brownian Motion) ---")
    print(f"Initial Value S0: {S0}")
    print(f"Empirical Expected Value E[S_T]: {S_T_empirical:.4f}")
    print(f"Theoretical Expected Value E[S_T]: {S_T_theoretical:.4f}")

    # Plot a sample of 20 simulated paths
    plt.figure(figsize=(10, 5))
    for i in range(min(20, num_paths)):
        plt.plot(t_values, S[i, :], lw=0.8, alpha=0.7)

    # Plot expected trajectory
    plt.plot(t_values, S0 * np.exp(mu * t_values), 'k--', label=r'Theoretical $E[S_t] = S_0 e^{\mu t}$', linewidth=2.5)

    plt.xlabel('Time (t)')
    plt.ylabel('Asset Price S(t)')
    plt.title(f'Monte Carlo Simulation of Geometric Brownian Motion ({num_paths} Paths)')
    plt.legend()
    plt.grid(True)
    plt.savefig('gbm_monte_carlo.png')
    plt.show()


if __name__ == "__main__":
    monte_carlo_gbm_simulation()
