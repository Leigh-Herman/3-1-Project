import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from joblib import Parallel, delayed


# Evolution Functions
def compute_hamiltonian_constraint(gamma_rr, K_rr):
    """
    Compute the Hamiltonian constraint:
    H = K_rr^2 / gamma_rr.
    """
    H = np.nan_to_num(K_rr ** 2 / gamma_rr)  # Avoid division by zero with nan_to_num
    return H


def monitor_constraints_parallel(gamma_rr, K_rr, n_jobs=-1):
    """
    Monitor the Hamiltonian constraint using parallelism.
    """

    def process_chunk(gamma_chunk, K_chunk):
        H = compute_hamiltonian_constraint(gamma_chunk, K_chunk)
        return np.max(np.abs(H))

    chunk_results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(gamma_rr[i, :], K_rr[i, :]) for i in range(gamma_rr.shape[0])
    )
    return max(chunk_results)


def evolve_lapse_1pluslog(alpha, K, dt):
    return alpha - 2 * alpha * K * dt


def evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt):
    return gamma_rr - 2 * alpha * K_rr * dt


def evolve_curvature(alpha, h_rr, K_rr, dt, dr):
    return K_rr - alpha * np.gradient(h_rr, dr, axis=0) * dt


# Parameters and Initialization
Nr = 50
Ntheta = 25
r_min, r_max = 1.0, 20.0
dt = 0.1
T_max = 2.0

r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(0.0, np.pi, Ntheta)
gamma_rr = np.ones((Nr, Ntheta))  # Metric component
K_rr = np.random.rand(Nr, Ntheta) * 0.1  # Extrinsic curvature (initial)
h_rr = np.random.rand(Nr, Ntheta) * 0.1  # Perturbations
alpha = np.ones(Nr)  # Lapse function
beta_r = np.zeros(Nr)  # Shift vector

# Create 2D Cartesian grid for visualization
r_mesh, theta_mesh = np.meshgrid(r, theta)
x = r_mesh * np.sin(theta_mesh)
y = r_mesh * np.cos(theta_mesh)

# Storage for Visualization
time_steps = int(T_max / dt)
H_data = np.zeros((time_steps, Nr, Ntheta))  # Store Hamiltonian constraints

# Time Evolution Loop with Parallel Stability Analysis and Visualization
time = 0.0
step = 0
while step < time_steps:
    # Evolve the lapse, metric, and curvature
    alpha = evolve_lapse_1pluslog(alpha, K_rr.mean(axis=1), dt)
    gamma_rr = evolve_metric(alpha[:, None], beta_r[:, None], K_rr, gamma_rr, dt)
    K_rr = evolve_curvature(alpha[:, None], h_rr, K_rr, dt, r[1] - r[0])

    # Monitor the Hamiltonian constraint in parallel
    max_H = monitor_constraints_parallel(gamma_rr, K_rr)
    print(f"Time: {time:.2f}, Max Hamiltonian Constraint (Parallel): {max_H:.5e}")

    # Store Hamiltonian constraint for visualization
    H_data[step, :, :] = compute_hamiltonian_constraint(gamma_rr, K_rr)

    # Increment time
    time += dt
    step += 1

# Visualization Setup
fig, ax = plt.subplots(figsize=(8, 6))
cbar = None


def update(frame):
    global cbar
    ax.clear()
    H = H_data[frame, :, :]
    contour = ax.contourf(x, y, H.T, levels=100, cmap="viridis")
    if cbar is None:
        cbar = plt.colorbar(contour, ax=ax, label="Hamiltonian Constraint")
    ax.set_title(f"Hamiltonian Constraint at Time {frame * dt:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# Create Animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=100)
ani.save("hamiltonian_constraint_evolution_with_loop.gif", writer="pillow")
plt.close(fig)