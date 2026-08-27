import numpy as np


def markov_chain_simulation(steps=10000):
    """Simulates a Discrete-Time Markov Chain (DTMC) with 3 states

    and computes the empirical vs. theoretical stationary distribution.
    """
    np.random.seed(42)

    # 1. Define Transition Probability Matrix P (3 states: 0, 1, 2)
    # P[i, j] = P(X_{n+1} = j | X_n = i)
    P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])

    states = [0, 1, 2]
    current_state = 0
    state_history = [current_state]

    # 2. Simulate State Transitions over Time
    for _ in range(steps):
        next_state = np.random.choice(states, p=P[current_state])
        state_history.append(next_state)
        current_state = next_state

    # 3. Calculate Empirical Stationary Distribution
    state_counts = np.bincount(state_history, minlength=len(states))
    empirical_pi = state_counts / len(state_history)

    # 4. Compute Theoretical Stationary Distribution pi * P = pi
    # Solving (P^T - I)pi^T = 0 with constraint sum(pi) = 1
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find eigenvector corresponding to eigenvalue = 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary_vector = np.real(eigenvectors[:, idx])
    theoretical_pi = stationary_vector / np.sum(stationary_vector)

    print("--- Discrete-Time Markov Chain Simulation ---")
    print(f"Empirical Stationary Distribution (after {steps} steps):")
    for s, prob in enumerate(empirical_pi):
        print(f" State {s}: {prob:.4f}")

    print("\nTheoretical Stationary Distribution:")
    for s, prob in enumerate(theoretical_pi):
        print(f" State {s}: {prob:.4f}")


if __name__ == "__main__":
    markov_chain_simulation()
