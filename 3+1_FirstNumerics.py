import numpy as np
import matplotlib.pyplot as plt

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
T_max = 2.0  # Maximum simulation time (reduced for faster testing)

# Create grid
r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)
phi = np.linspace(phi_min, phi_max, Nphi)
dr = r[1] - r[0]

# Initialize lapse, shift, and metric components
alpha = np.ones((Nr, Ntheta, Nphi))  # Initial lapse
beta_r = -np.sqrt(2 * M1 / r[:, None, None])  # Initial shift (radial)
gamma_rr = np.ones((Nr, Ntheta, Nphi))  # Initial radial metric component
gamma_theta_theta = (r[:, None, None] ** 2) * np.ones((Nr, Ntheta, Nphi))
gamma_phi_phi = gamma_theta_theta * (np.sin(theta)[None, :, None] ** 2)

# Extrinsic curvature components
K_rr = np.zeros((Nr, Ntheta, Nphi))
K_theta_theta = np.zeros((Nr, Ntheta, Nphi))
K_phi_phi = np.zeros((Nr, Ntheta, Nphi))

# Perturbation components
h_rr = np.zeros((Nr, Ntheta, Nphi))
h_theta_theta = np.zeros((Nr, Ntheta, Nphi))
h_phi_phi = np.zeros((Nr, Ntheta, Nphi))


# Evolution equations
def evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt):
    return gamma_rr - 2 * alpha * K_rr * dt


def evolve_curvature(alpha, h_rr, K_rr, dt):
    return K_rr - alpha * np.gradient(h_rr, dr, axis=0) * dt


# Hamiltonian constraint
def hamiltonian_constraint(K_rr, gamma_rr):
    return np.nan_to_num((K_rr ** 2) / gamma_rr)


# Simulation loop
time = 0.0
frames = []  # To store visualization data for animation
while time < T_max:
    # Update metric
    gamma_rr = evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt)

    # Update extrinsic curvature
    K_rr = evolve_curvature(alpha, h_rr, K_rr, dt)

    # Compute the Hamiltonian constraint
    H = hamiltonian_constraint(K_rr, gamma_rr)

    # Log progress
    if int(time / dt) % 10 == 0:
        print(f"Time: {time:.2f}, Max Hamiltonian Constraint: {np.max(np.abs(H))}")

    # Store a 2D slice for visualization (e.g., phi=0 plane)
    frames.append(H[:, :, 0])

    # Increment time
    time += dt


# Visualization of the Hamiltonian constraint evolution
def visualize_evolution(frames, r, theta):
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    x = r_mesh * np.sin(theta_mesh)
    y = r_mesh * np.cos(theta_mesh)

    for i, frame in enumerate(frames):
        plt.figure(figsize=(8, 6))
        plt.contourf(x, y, frame.T, levels=100, cmap="viridis")
        plt.colorbar(label="Hamiltonian Constraint (H)")
        plt.title(f"Hamiltonian Constraint at Time {i * dt:.2f}")
        plt.xlabel("x (Cartesian)")
        plt.ylabel("y (Cartesian)")
        plt.show()


# Visualize the evolution
visualize_evolution(frames, r, theta)
