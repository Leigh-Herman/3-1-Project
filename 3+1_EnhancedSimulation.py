import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.special import sph_harm
from scipy.interpolate import interp1d
from matplotlib.animation import PillowWriter, FuncAnimation

# Utility Functions
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


def refine_mesh(h, r, threshold):
    """
    Adaptive Mesh Refinement (AMR).
    """
    gradients = np.abs(np.gradient(h, axis=0))
    refined_r = [r[0]]
    refined_h = [h[0, :]]

    for i in range(1, len(r)):
        refined_r.append(r[i])
        refined_h.append(h[i, :])
        if np.max(gradients[i - 1, :]) > threshold:
            mid_r = 0.5 * (r[i - 1] + r[i])
            mid_h = 0.5 * (h[i - 1, :] + h[i, :])
            refined_r.insert(-1, mid_r)
            refined_h.insert(-1, mid_h)

    return np.array(refined_r), np.array(refined_h)


def interpolate_field(field, old_r, new_r):
    """
    Interpolate a field onto a refined mesh.
    """
    interpolated_field = np.zeros((len(new_r), field.shape[1]))
    for i in range(field.shape[1]):
        interp_func = interp1d(old_r, field[:, i], kind='linear', fill_value="extrapolate")
        interpolated_field[:, i] = interp_func(new_r)
    return interpolated_field


# Evolution Functions
def evolve_lapse_1pluslog(alpha, K, dt):
    return alpha - 2 * alpha * K * dt


def evolve_shift_gammadriver(beta, B, Gamma, eta, dt):
    B_new = B + dt * (Gamma - eta * B)
    beta_new = beta + (3 / 4) * B_new * dt
    return beta_new, B_new


def evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt):
    return gamma_rr - 2 * alpha * K_rr * dt


def evolve_curvature(alpha, h_rr, K_rr, dt, dr):
    return K_rr - alpha * np.gradient(h_rr, dr, axis=0) * dt


# Boundary Conditions
def apply_outgoing_wave_boundary(h, dr, dt):
    h[0, :] = h[1, :]
    h[-1, :] = h[-2, :] - (dt / dr) * (h[-2, :] - h[-3, :])
    return h


def apply_inner_boundary_condition(h):
    h[0, :] = 0
    return h


# Perturbation Initialization Using Spherical Harmonics
def initialize_perturbation_spherical(r, theta, l, m, amplitude):
    Y_lm = sph_harm(m, l, 0, theta).real
    return amplitude * np.outer(r, Y_lm)


# Visualization Functions
def create_visualizations(H_data, x, y, dt, time_steps):
    fig, ax = plt.subplots(figsize=(8, 6))
    cbar = None

    def update(frame):
        nonlocal cbar
        ax.clear()
        H = H_data[frame, :, :]
        contour = ax.contourf(x, y, H.T, levels=100, cmap="viridis")
        if cbar is None:
            cbar = plt.colorbar(contour, ax=ax, label="Hamiltonian Constraint")
        ax.set_title(f"Hamiltonian Constraint at Time {frame * dt:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ani = FuncAnimation(fig, update, frames=time_steps, interval=100)
    ani.save("full_simulation_evolution.gif", writer="pillow")
    plt.close(fig)


# Parameters and Initialization
Nr = 50
Ntheta = 25
r_min, r_max = 1.0, 20.0
theta_min, theta_max = 0.0, np.pi
l, m = 2, 0
amplitude = 0.1
dt = 0.1
T_max = 2.0
threshold = 0.2

r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)
r_mesh, theta_mesh = np.meshgrid(r, theta)
x = r_mesh * np.sin(theta_mesh)
y = r_mesh * np.cos(theta_mesh)

gamma_rr = np.ones((Nr, Ntheta))
K_rr = np.random.rand(Nr, Ntheta) * 0.1
h_rr = initialize_perturbation_spherical(r, theta, l, m, amplitude)
alpha = np.ones(Nr)
beta_r = np.zeros(Nr)
time_steps = int(T_max / dt)
H_data = np.zeros((time_steps, Nr, Ntheta))

# Evolution Loop
time = 0.0
for step in range(time_steps):
    alpha = evolve_lapse_1pluslog(alpha, K_rr.mean(axis=1), dt)
    beta_r, _ = evolve_shift_gammadriver(beta_r, np.zeros_like(beta_r), np.zeros_like(beta_r), 2.0, dt)
    gamma_rr = evolve_metric(alpha[:, None], beta_r[:, None], K_rr, gamma_rr, dt)
    K_rr = evolve_curvature(alpha[:, None], h_rr, K_rr, dt, r[1] - r[0])
    h_rr = apply_outgoing_wave_boundary(h_rr, r[1] - r[0], dt)
    h_rr = apply_inner_boundary_condition(h_rr)

    old_r = r
    r, h_rr = refine_mesh(h_rr, r, threshold)
    gamma_rr = interpolate_field(gamma_rr, old_r, r)
    K_rr = interpolate_field(K_rr, old_r, r)

    max_H = monitor_constraints_parallel(gamma_rr, K_rr)
    print(f"Time: {time:.2f}, Max Hamiltonian Constraint: {max_H:.5e}")
    H_data[step, :, :] = compute_hamiltonian_constraint(gamma_rr, K_rr)
    time += dt

def create_visualizations_pillow_revised(H_data, x, y, dt, time_steps):
    """
    Create animated visualization (GIF) of Hamiltonian constraint evolution using Pillow writer.
    Args:
        H_data: 3D array of Hamiltonian constraint values (time_steps x Nr x Ntheta).
        x: 2D Cartesian x-coordinates for visualization.
        y: 2D Cartesian y-coordinates for visualization.
        dt: Time step size.
        time_steps: Total number of timesteps.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cbar = None

    def update(frame):
        nonlocal cbar
        ax.clear()
        H = H_data[frame, :, :]
        contour = ax.contourf(x, y, H.T, levels=100, cmap="viridis")
        if cbar is None:
            cbar = plt.colorbar(contour, ax=ax, label="Hamiltonian Constraint")
        ax.set_title(f"Hamiltonian Constraint at Time {frame * dt:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Creating and saving the animation using PillowWriter
    ani = FuncAnimation(fig, update, frames=time_steps, interval=100)
    pillow_writer = PillowWriter(fps=10)
    ani.save("hamiltonian_constraint_EnhancedEvolution_pillow.gif", writer=pillow_writer)
    plt.close(fig)
    print("Visualization saved as hamiltonian_constraint_evolution_pillow.gif")


# Call the visualization function
create_visualizations_pillow_revised(H_data, x, y, dt, time_steps)
