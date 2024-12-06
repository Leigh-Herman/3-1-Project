import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Constants
M1 = 1.0  # Mass of the primary black hole
M2 = 0.01  # Mass of the secondary black hole
r_min, r_max = 1.0, 20.0  # Radial domain
theta_min, theta_max = 0.0, np.pi  # Angular domain
phi_min, phi_max = 0.0, 2 * np.pi  # Azimuthal domain

# Grid resolution
Nr = 100  # Number of radial points
Ntheta = 50  # Number of angular points
Nphi = 50  # Number of azimuthal points
dt = 0.01  # Time step
T_max = 2.0  # Shortened for testing

# Create grid
r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)
phi = np.linspace(phi_min, phi_max, Nphi)
dr = r[1] - r[0]

# Initialize arrays
def evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt):
    return gamma_rr - 2 * alpha * K_rr * dt

def evolve_curvature(alpha, h_rr, K_rr, dt):
    return K_rr - alpha * np.gradient(h_rr, dr, axis=0) * dt

def hamiltonian_constraint(K_rr, gamma_rr):
    return np.nan_to_num((K_rr**2) / gamma_rr)

# Function to evolve and prepare animation data
def evolve_and_prepare_animation_data(T_max, dt, Nr, Ntheta, Nphi, r, theta):
    gamma_rr = np.ones((Nr, Ntheta, Nphi))
    K_rr = np.zeros((Nr, Ntheta, Nphi))
    h_rr = np.zeros((Nr, Ntheta, Nphi))
    alpha = np.ones((Nr, Ntheta, Nphi))  # Lapse function

    H_slices = []
    time = 0.0
    while time < T_max:
        gamma_rr = evolve_metric(alpha, -np.sqrt(2 * M1 / r[:, None, None]), K_rr, gamma_rr, dt)
        K_rr = evolve_curvature(alpha, h_rr, K_rr, dt)
        H = hamiltonian_constraint(K_rr, gamma_rr)
        H_slices.append(H[:, :, 0])  # Store phi=0 slice
        time += dt
    return H_slices

# Prepare the animation data
H_slices = evolve_and_prepare_animation_data(T_max, dt, Nr, Ntheta, Nphi, r, theta)

# Visualization setup
r_mesh, theta_mesh = np.meshgrid(r, theta)
x = r_mesh * np.sin(theta_mesh)
y = r_mesh * np.cos(theta_mesh)

fig, ax = plt.subplots(figsize=(8, 6))
cbar = None

def update(frame):
    global cbar
    ax.clear()
    H_data = H_slices[frame]
    contour = ax.contourf(x, y, H_data.T, levels=100, cmap="viridis")
    if cbar is None:
        cbar = plt.colorbar(contour, ax=ax, label="Hamiltonian Constraint (H)")
    ax.set_title(f"Hamiltonian Constraint (Time={frame*dt:.2f})")
    ax.set_xlabel("x (Cartesian)")
    ax.set_ylabel("y (Cartesian)")

# Create the animation with Pillow Writer
ani = FuncAnimation(fig, update, frames=len(H_slices), interval=50)
gif_path = 'hamiltonian_constraint_evolution_pillow.gif'
ani.save(gif_path, writer=PillowWriter(fps=20))
plt.close(fig)
gif_path
