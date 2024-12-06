import numpy as np
from scipy.special import sph_harm
from scipy.interpolate import interp1d

# Initialize Perturbation with Spherical Harmonics
def initialize_perturbation_spherical(r, theta, l, m, amplitude):
    """
    Initialize the perturbation using spherical harmonics.
    """
    Y_lm = sph_harm(m, l, 0, theta).real  # Spherical harmonic with phi=0 (2D axisymmetric case)
    perturbation = amplitude * np.outer(r, Y_lm)
    return perturbation

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

# Adaptive Mesh Refinement and Interpolation
def refine_mesh(h, r, threshold):
    gradients = np.abs(np.gradient(h, axis=0))
    refined_r = [r[0]]  # Start with the first radial point
    refined_h = [h[0, :]]  # Start with the first row of h

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
    interpolated_field = np.zeros((len(new_r), field.shape[1]))
    for i in range(field.shape[1]):
        interp_func = interp1d(old_r, field[:, i], kind='linear', fill_value="extrapolate")
        interpolated_field[:, i] = interp_func(new_r)
    return interpolated_field

# Initialize Parameters and Grid
Nr = 20
Ntheta = 10
r_min, r_max = 1.0, 20.0
theta_min, theta_max = 0.0, np.pi
threshold = 0.2
dt = 0.01
T_max = 1.0
l = 2  # Degree of spherical harmonic
m = 0  # Order of spherical harmonic
amplitude = 0.1  # Amplitude of the perturbation

r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)

# Initialize Perturbation Using Spherical Harmonics
h_rr = initialize_perturbation_spherical(r, theta, l, m, amplitude)

# Initialize Other Arrays
gamma_rr = np.ones((Nr, Ntheta))
K_rr = np.zeros((Nr, Ntheta))
alpha = np.ones(Nr)
beta = np.zeros(Nr)
B = np.zeros(Nr)
Gamma = np.random.rand(Nr) * 0.01
eta = 2.0

# Evolution Loop
time = 0.0
while time < T_max:
    alpha = evolve_lapse_1pluslog(alpha, K_rr.mean(axis=1), dt)
    beta, B = evolve_shift_gammadriver(beta, B, Gamma, eta, dt)
    gamma_rr = evolve_metric(alpha[:, None], beta[:, None], K_rr, gamma_rr, dt)
    K_rr = evolve_curvature(alpha[:, None], h_rr, K_rr, dt, r[1] - r[0])
    h_rr = apply_outgoing_wave_boundary(h_rr, r[1] - r[0], dt)
    h_rr = apply_inner_boundary_condition(h_rr)
    old_r = r
    r, h_rr = refine_mesh(h_rr, r, threshold)
    gamma_rr = interpolate_field(gamma_rr, old_r, r)
    K_rr = interpolate_field(K_rr, old_r, r)
    time += dt

# Final Output for Verification
print("Final Perturbation Shape:", h_rr.shape)
print("Final Radial Grid:", r)
