import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d
from scipy.special import sph_harm

# Utility Functions
def compute_hamiltonian_constraint(gamma_rr, K_rr):
    """
    Compute the Hamiltonian constraint:
    H = K_rr^2 / gamma_rr.
    """
    # Ensure gamma_rr and K_rr are 2D
    if gamma_rr.ndim != 2 or K_rr.ndim != 2:
        raise ValueError(f"Inputs must be 2D, got {gamma_rr.ndim}D and {K_rr.ndim}D.")

    # Avoid division by zero with np.nan_to_num
    H = np.nan_to_num(K_rr**2 / gamma_rr)
    return H  # Should return shape (100, 50)

def evolve_metric(alpha_broadcasted, beta_r, K_rr, gamma_rr, dt):
    """
    Evolve the metric component gamma_rr.
    """
    return gamma_rr - 2 * alpha_broadcasted * K_rr * dt

def evolve_curvature(alpha, h_rr, K_rr, dt, dr):
    """
    Evolve the extrinsic curvature K_rr.
    """
    # Compute the gradient and ensure the shape of the output matches expectations
    grad_h_rr = np.gradient(h_rr, dr, axis=0)
    print(f"grad_h_rr shape: {grad_h_rr.shape}")  # Debugging shape of gradient
    return K_rr - alpha[:, None] * grad_h_rr * dt

def apply_outgoing_wave_boundary(h, dr, dt):
    """
    Apply outgoing wave boundary conditions.
    """
    h[0, :] = h[1, :]
    h[-1, :] = h[-2, :] - (dt / dr) * (h[-2, :] - h[-3, :])
    return h

def initialize_perturbation_spherical(r, theta, l, m, amplitude):
    """
    Initialize perturbation using spherical harmonics.
    """
    Y_lm = sph_harm(m, l, 0, theta).real
    return amplitude * np.outer(r, Y_lm)

def create_visualizations_pillow(H_data, x, y, dt, time_steps, filename="plunging_black_hole.gif"):
    """
    Create animated visualization of Hamiltonian constraint evolution.
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

    ani = FuncAnimation(fig, update, frames=time_steps, interval=100)
    ani.save(filename, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Visualization saved as {filename}")

# Simulation Parameters
Nr = 100
Ntheta = 50
r_min, r_max = 2.0, 50.0
theta_min, theta_max = 0.0, np.pi
l, m = 2, 0
amplitude = 0.1
dt = 0.05
T_max = 10.0

M_large = 1.0  # Mass of the large black hole
M_small = 0.1  # Mass of the small black hole
r_small_initial = 10.0  # Initial position of the small black hole
velocity_small = -0.5  # Radial velocity of the small black hole

# Grid Initialization
r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)
r_mesh, theta_mesh = np.meshgrid(r, theta)
x = r_mesh * np.sin(theta_mesh)
y = r_mesh * np.cos(theta_mesh)

# Initial Conditions
gamma_rr = np.ones((Nr, Ntheta)) * (1 + 2 * M_large / r[:, None])
K_rr = np.zeros((Nr, Ntheta))  # Ensure 2D shape
print(f"Initial K_rr shape: {K_rr.shape}")  # Expected: (Nr, Ntheta)
h_rr = initialize_perturbation_spherical(r, theta, l, m, amplitude)
alpha = np.ones(Nr)
time_steps = int(T_max / dt)

# Ensure `H_data` has the correct shape
H_data = np.zeros((time_steps, Nr, Ntheta))  # Shape: (time_steps, 100, 50)
print(f"H_data shape: {H_data.shape}")  # Should be (time_steps, Nr, Ntheta)

# Evolution Loop
time = 0.0
for step in range(time_steps):
    print(f"Step {step}: Starting evolution")

    # Debugging shapes
    print(f"alpha shape: {alpha.shape}")  # Should be (Nr,)
    print(f"gamma_rr shape: {gamma_rr.shape}")  # Should be (Nr, Ntheta)
    print(f"K_rr shape: {K_rr.shape}")  # Should be (Nr, Ntheta)

    # Broadcast alpha to match gamma_rr
    alpha = alpha.reshape(Nr)  # Ensure alpha is 1D
    alpha_broadcasted = np.broadcast_to(alpha[:, None], gamma_rr.shape)  # Explicit broadcasting
    print(f"alpha_broadcasted shape: {alpha_broadcasted.shape}")  # Should be (Nr, Ntheta)

    # Evolve metric
    gamma_rr = evolve_metric(alpha_broadcasted, np.zeros_like(alpha_broadcasted), K_rr, gamma_rr, dt)

    # Update curvature
    K_rr = evolve_curvature(alpha_broadcasted, h_rr, K_rr, dt, r[1] - r[0])
    if K_rr.ndim == 3:
        K_rr = K_rr.squeeze()  # Remove any singleton dimensions

    # Add effects of the plunging small black hole
    r_small_current = r_small_initial + step * velocity_small * dt
    if r_small_current < 2 * M_large:  # Terminate simulation if the small black hole merges
        print("Small black hole has merged with the large black hole.")
        break
    gamma_rr += M_small / np.maximum(r[:, None] - r_small_current, 1e-6)**2

    # Apply boundary conditions
    h_rr = apply_outgoing_wave_boundary(h_rr, r[1] - r[0], dt)
    if h_rr.ndim != 2:
        h_rr = h_rr.reshape(Nr, Ntheta)

    # Compute and store Hamiltonian constraint
    H = compute_hamiltonian_constraint(gamma_rr, K_rr)
    print(f"H shape: {H.shape}")  # Should be (100, 50)
    H_data[step, :, :] = H

    # Increment time
    time += dt


# Visualization
create_visualizations_pillow(H_data, x, y, dt, step, filename="plunging_black_hole.gif")
