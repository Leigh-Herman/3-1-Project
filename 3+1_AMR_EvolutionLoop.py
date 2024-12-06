import numpy as np
from scipy.interpolate import interp1d


# Adaptive Mesh Refinement (AMR)
def refine_mesh(h, r, threshold):
    """
    Refine the mesh by adding additional points in regions where the gradient of h exceeds a threshold.
    """
    gradients = np.abs(np.gradient(h, axis=0))
    refined_r = [r[0]]  # Start with the first radial point
    refined_h = [h[0, :]]  # Start with the first row of h

    for i in range(1, len(r)):
        refined_r.append(r[i])
        refined_h.append(h[i, :])
        if np.max(gradients[i - 1, :]) > threshold:
            # Add an intermediate point
            mid_r = 0.5 * (r[i - 1] + r[i])
            mid_h = 0.5 * (h[i - 1, :] + h[i, :])
            refined_r.insert(-1, mid_r)
            refined_h.insert(-1, mid_h)

    return np.array(refined_r), np.array(refined_h)


# Field Interpolation
def interpolate_field(field, old_r, new_r):
    """
    Interpolates a field onto a new radial grid.
    """
    interpolated_field = np.zeros((len(new_r), field.shape[1]))
    for i in range(field.shape[1]):  # Interpolate along the radial dimension for each angular slice
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
    h[0, :] = h[1, :]  # Inner edge
    h[-1, :] = h[-2, :] - (dt / dr) * (h[-2, :] - h[-3, :])  # Outer edge
    return h


def apply_inner_boundary_condition(h):
    h[0, :] = 0  # Freeze perturbation at the inner boundary
    return h


# Parameters and Grid Initialization
Nr = 20  # Initial radial points
Ntheta = 10  # Angular points
r_min, r_max = 1.0, 20.0  # Radial domain
theta_min, theta_max = 0.0, np.pi  # Angular domain
threshold = 0.2  # Refinement threshold
dt = 0.01  # Time step
T_max = 1.0  # Maximum simulation time

r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)

# Initialize Arrays
gamma_rr = np.ones((Nr, Ntheta))
K_rr = np.zeros((Nr, Ntheta))
h_rr = np.random.rand(Nr, Ntheta) * 0.1  # Initial perturbations
alpha = np.ones(Nr)  # Initial lapse function
beta = np.zeros(Nr)  # Initial shift vector
B = np.zeros(Nr)  # Auxiliary variable for shift
Gamma = np.random.rand(Nr) * 0.01  # Proxy for Christoffel symbols
eta = 2.0  # Damping parameter

# Time Evolution Loop with AMR
time = 0.0
while time < T_max:
    # Evolve lapse using 1+log slicing
    alpha = evolve_lapse_1pluslog(alpha, K_rr.mean(axis=1), dt)

    # Evolve shift vector using Gamma-driver condition
    beta, B = evolve_shift_gammadriver(beta, B, Gamma, eta, dt)

    # Evolve metric and curvature
    gamma_rr = evolve_metric(alpha[:, None], beta[:, None], K_rr, gamma_rr, dt)
    K_rr = evolve_curvature(alpha[:, None], h_rr, K_rr, dt, r[1] - r[0])

    # Apply boundary conditions
    h_rr = apply_outgoing_wave_boundary(h_rr, r[1] - r[0], dt)
    h_rr = apply_inner_boundary_condition(h_rr)

    # Refine the mesh if needed
    old_r = r
    r, h_rr = refine_mesh(h_rr, r, threshold)
    gamma_rr = interpolate_field(gamma_rr, old_r, r)
    K_rr = interpolate_field(K_rr, old_r, r)

    # Increment time
    time += dt

# Final Output for Verification
print("Refined Radial Grid:", r)
print("Final Perturbation Array Shape:", h_rr.shape)
