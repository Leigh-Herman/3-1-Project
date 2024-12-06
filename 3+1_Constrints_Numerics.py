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
T_max = 10.0  # Maximum simulation time

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


# Evolution equations for the metric
def evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt):
    return gamma_rr - 2 * alpha * K_rr * dt


# Evolution equations for the extrinsic curvature
def evolve_curvature(alpha, h_rr, K_rr, dt):
    return K_rr - alpha * np.gradient(h_rr, dr, axis=0) * dt


# Hamiltonian constraint
def hamiltonian_constraint(K_rr, gamma_rr):
    return np.nan_to_num((K_rr ** 2) / gamma_rr)


# Time evolution
time = 0.0
while time < T_max:
    # Update metric
    gamma_rr = evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt)

    # Update extrinsic curvature
    K_rr = evolve_curvature(alpha, h_rr, K_rr, dt)

    # Compute Hamiltonian constraint
    H = hamiltonian_constraint(K_rr, gamma_rr)

    # Print or log the maximum constraint value for stability checks
    if int(time / dt) % 10 == 0:
        print(f"Time: {time}, Hamiltonian Constraint (max): {np.max(np.abs(H))}")

    # Increment time
    time += dt


# Visualization of the Hamiltonian constraint (final state)
def visualize_hamiltonian(H, r, theta):
    """
    Visualizes the Hamiltonian constraint as a heatmap.
    Args:
        H: Hamiltonian constraint values (numerical array).
        r: Radial coordinates.
        theta: Angular coordinates.
    """
    # Create a 2D grid for visualization (slice through azimuthal direction, e.g., phi=0)
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    H_slice = H[:, :, 0]  # Slice through the phi=0 plane

    # Convert spherical to Cartesian coordinates for plotting
    x = r_mesh * np.sin(theta_mesh)
    y = r_mesh * np.cos(theta_mesh)

    # Plot the Hamiltonian constraint as a 2D heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, H_slice.T, levels=100, cmap="viridis")
    plt.colorbar(label="Hamiltonian Constraint (H)")
    plt.title("Hamiltonian Constraint (Slice at $\phi=0$)")
    plt.xlabel("x (Cartesian)")
    plt.ylabel("y (Cartesian)")
    plt.show()


# Visualize the final Hamiltonian constraint
visualize_hamiltonian(H, r, theta)
