import numpy as np


def martingale_stopping_time_simulation(num_simulations=10000, barrier=10):
    """Simulates a symmetric random walk M_n (which is a Martingale) and verifies

    Doob's Optional Stopping Theorem for the stopping time tau = min{n : |M_n| = barrier}.
    """
    np.random.seed(42)
    stopped_values = []
    stopping_times = []

    for _ in range(num_simulations):
        M_n = 0
        n = 0
        
        # Random walk loop until stopping time tau
        while abs(M_n) < barrier:
            step = np.random.choice([-1, 1])
            M_n += step
            n += 1
            
        stopped_values.append(M_n)
        stopping_times.append(n)

    empirical_E_M_tau = np.mean(stopped_values)
    empirical_E_tau = np.mean(stopping_times)

    print("--- Martingales & Doob's Optional Stopping Theorem ---")
    print(f"Initial Value M_0: 0")
    print(f"Empirical E[M_tau]: {empirical_E_M_tau:.4f} (Theoretical E[M_tau] = M_0 = 0)")
    print(f"Empirical Expected Stopping Time E[tau]: {empirical_E_tau:.2f} (Theoretical E[tau] = barrier^2 = {barrier**2})")


if __name__ == "__main__":
    martingale_stopping_time_simulation()
