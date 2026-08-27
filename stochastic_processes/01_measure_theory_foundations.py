import numpy as np


def conditional_expectation_simulation(num_samples=100000):
    """Simulates conditional expectation E[X | G] where G is a partition of Omega.

    Omega is divided into two events: G1 = {X > 0} and G2 = {X <= 0}.
    """
    # 1. Define Random Variable X following a Normal Distribution N(0, 1)
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=num_samples)

    # 2. Define Partition G (sigma-algebra) based on conditions
    condition = X > 0 # Event G1: X > 0, Event G2: X <= 0

    # 3. Calculate Theoretical & Empirical Conditional Expectation E[X | X > 0]
    e_x_given_g1 = np.mean(X[condition])
    e_x_given_g2 = np.mean(X[~condition])

    # Construct the random variable E[X | G]
    E_X_given_G = np.where(condition, e_x_given_g1, e_x_given_g2)

    print("--- Measure Theory & Conditional Expectation ---")
    print(f"Overall Expectation E[X]: {np.mean(X):.4f}")
    print(f"Conditional Expectation E[X | X > 0]: {e_x_given_g1:.4f}")
    print(f"Conditional Expectation E[X | X <= 0]: {e_x_given_g2:.4f}")

    # Law of Total Expectation: E[E[X|G]] = E[X]
    print(
        f"Verification of Total Expectation E[E[X|G]]: {np.mean(E_X_given_G):.4f}"
    )


if __name__ == "__main__":
    conditional_expectation_simulation()
