import numpy as np


def ito_integral_simulation(T=1.0, N=10000, num_simulations=50000):
    """Simulates the Itô integral \int_0^T B_t dB_t using Euler-Maruyama discretization

    and verifies Itô's Lemma formula: 0.5 * B_T^2 - 0.5 * T.
    """
    np.random.seed(42)
    dt = T / N

    # Generate Brownian increments dB ~ N(0, dt) for all paths
    dB = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_simulations, N))

    # Construct Brownian paths B_t (excluding initial 0 for integration loop)
    B = np.zeros((num_simulations, N + 1))
    B[:, 1:] = np.cumsum(dB, axis=1)

    # Compute Itô Integral: sum(B_{t_i} * (B_{t_{i+1}} - B_{t_i}))
    # Left-point approximation (Itô convention)
    ito_integrals = np.sum(B[:, :-1] * dB, axis=1)

    # Compute Theoretical Value from Itô's Lemma: 0.5 * B_T^2 - 0.5 * T
    B_T = B[:, -1]
    ito_theoretical = 0.5 * (B_T**2) - 0.5 * T

    print("--- Stochastic Calculus & Itô's Lemma Verification ---")
    print(
        f"Empirical Mean of Itô Integral: {np.mean(ito_integrals):.4f} (Theoretical E[\int B dB] = 0)"
    )
    print(f"Mean Absolute Error vs Itô Lemma Formula: {np.mean(np.abs(ito_integrals - ito_theoretical)):.6f}")


if __name__ == "__main__":
    ito_integral_simulation()
