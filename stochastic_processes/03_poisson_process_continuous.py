import numpy as np
import matplotlib.pyplot as plt


def poisson_process_simulation(lambda_rate=2.0, T=10.0, num_paths=3):
    """
    Simulates paths of a Homogeneous Poisson Process N(t) with rate lambda over time [0, T].
    """
    np.random.seed(42)
    plt.figure(figsize=(10, 5))

    for path in range(num_paths):
        arrival_times = [0.0]
        current_time = 0.0

        # Generate inter-arrival times using Exponential Distribution Exp(lambda)
        while current_time < T:
            inter_arrival = np.random.exponential(1.0 / lambda_rate)
            current_time += inter_arrival
            if current_time <= T:
                arrival_times.append(current_time)

        # Event counts N(t)
        event_counts = np.arange(len(arrival_times))

        # Plot step function for Poisson Process path
        plt.step(arrival_times, event_counts, where='post', label=f'Path {path + 1}')

    # Plot theoretical expected value E[N(t)] = lambda * t
    t_vals = np.linspace(0, T, 100)
    plt.plot(t_vals, lambda_rate * t_vals, 'k--', label=r'Theoretical $E[N(t)] = \lambda t$', linewidth=2)

    plt.xlabel('Time (t)')
    plt.ylabel('Number of Events N(t)')
    plt.title(f'Homogeneous Poisson Process Paths (Rate $\lambda = {lambda_rate}$)')
    plt.legend()
    plt.grid(True)
    plt.savefig('poisson_process.png')
    plt.show()


if __name__ == "__main__":
    poisson_process_simulation()
