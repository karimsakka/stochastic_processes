import numpy as np
import matplotlib.pyplot as plt


def brownian_motion_simulation(T=1.0, N=1000, num_paths=5):
    """
    Simulates paths of Standard Brownian Motion B(t) over [0, T]
    and computes the Empirical Quadratic Variation sum((B_{t_i} - B_{t_{i-1}})^2).
    """
    np.random.seed(42)
    dt = T / N
    t_values = np.linspace(0, T, N + 1)

    plt.figure(figsize=(10, 5))

    for path in range(num_paths):
        # Generate independent standard normal increments dB ~ N(0, dt)
        dB = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=N)
        
        # Construct Brownian Motion path B_t = sum(dB) with B_0 = 0
        B = np.insert(np.cumsum(dB), 0, 0.0)

        # Calculate Quadratic Variation for this path
        quad_var = np.sum((np.diff(B)) ** 2)

        plt.plot(t_values, B, label=f'Path {path + 1} (Quad Var: {quad_var:.3f})')

    plt.xlabel('Time (t)')
    plt.ylabel('B(t)')
    plt.title(f'Standard Brownian Motion Paths & Quadratic Variation (Theoretical Quad Var = T = {T})')
    plt.legend()
    plt.grid(True)
    plt.savefig('brownian_motion.png')
    plt.show()


if __name__ == "__main__":
    brownian_motion_simulation()
import numpy as np
import matplotlib.pyplot as plt


def brownian_motion_simulation(T=1.0, N=1000, num_paths=5):
    """
    Simulates paths of Standard Brownian Motion B(t) over [0, T]
    and computes the Empirical Quadratic Variation sum((B_{t_i} - B_{t_{i-1}})^2).
    """
    np.random.seed(42)
    dt = T / N
    t_values = np.linspace(0, T, N + 1)

    plt.figure(figsize=(10, 5))

    for path in range(num_paths):
        # Generate independent standard normal increments dB ~ N(0, dt)
        dB = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=N)
        
        # Construct Brownian Motion path B_t = sum(dB) with B_0 = 0
        B = np.insert(np.cumsum(dB), 0, 0.0)

        # Calculate Quadratic Variation for this path
        quad_var = np.sum((np.diff(B)) ** 2)

        plt.plot(t_values, B, label=f'Path {path + 1} (Quad Var: {quad_var:.3f})')

    plt.xlabel('Time (t)')
    plt.ylabel('B(t)')
    plt.title(f'Standard Brownian Motion Paths & Quadratic Variation (Theoretical Quad Var = T = {T})')
    plt.legend()
    plt.grid(True)
    plt.savefig('brownian_motion.png')
    plt.show()


if __name__ == "__main__":
    brownian_motion_simulation()
