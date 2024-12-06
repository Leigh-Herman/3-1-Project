import numpy as np

# Constants
M1 = 1.0  # Mass of the primary black hole


# Evolution Functions
def evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt):
    """
    Evolve the metric component gamma_rr using simplified evolution.
    """
    return gamma_rr - 2 * alpha * K_rr * dt


def evolve_curvature(alpha, h_rr, K_rr, dt, dr):
    """
    Evolve the extrinsic curvature component K_rr.
    """
    return K_rr - alpha * np.gradient(h_rr, dr, axis=0) * dt


def hamiltonian_constraint(K_rr, gamma_rr):
    """
    Compute the Hamiltonian constraint for monitoring stability.
    """
    return np.nan_to_num((K_rr ** 2) / gamma_rr)


# Boundary Conditions
def apply_outgoing_wave_boundary(h, dr, dt):
    """
    Apply outgoing wave boundary condition at the edges of the grid.
    """
    h[0, :] = h[1, :]  # Inner edge (reflection avoided)
    h[-1, :] = h[-2, :] - (dt / dr) * (h[-2, :] - h[-3, :])  # Outer edge
    return h


def apply_inner_boundary_condition(h):
    """
    Apply inner boundary condition near the black hole horizon.
    """
    h[0, :] = 0  # Freeze perturbation at the inner boundary
    return h


# Initialize Parameters and Grid
Nr = 100  # Radial points
Ntheta = 50  # Angular points
r_min, r_max = 1.0, 20.0  # Radial domain
theta_min, theta_max = 0.0, np.pi  # Angular domain
dr = (r_max - r_min) / Nr  # Radial step size
dt = 0.01  # Time step
T_max = 2.0  # Maximum simulation time

r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)

# Initialize arrays
gamma_rr = np.ones((Nr, Ntheta))
K_rr = np.zeros((Nr, Ntheta))
h_rr = np.random.rand(Nr, Ntheta) * 0.1  # Initial small random perturbations
alpha = np.ones((Nr, Ntheta))  # Lapse function
beta_r = -np.sqrt(2 * M1 / r)[:, None]  # Shift vector

# Time Evolution Loop
time = 0.0
while time < T_max:
    # Evolve metric and curvature
    gamma_rr = evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt)
    K_rr = evolve_curvature(alpha, h_rr, K_rr, dt, dr)

    # Apply boundary conditions
    h_rr = apply_outgoing_wave_boundary(h_rr, dr, dt)
    h_rr = apply_inner_boundary_condition(h_rr)

    # Compute and monitor constraints
    H = hamiltonian_constraint(K_rr, gamma_rr)
    if time % 0.1 < dt:  # Log every 0.1 seconds
        print(f"Time: {time:.2f}, Max Hamiltonian Constraint: {np.max(H):.5f}")

    # Increment time
    time += dt
