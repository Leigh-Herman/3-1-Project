import numpy as np


def compute_hamiltonian_constraint(gamma_rr, K_rr):
    """
    Compute the Hamiltonian constraint:
    H = K_rr^2 / gamma_rr.
    """
    H = np.nan_to_num(K_rr ** 2 / gamma_rr)  # Avoid division by zero with nan_to_num
    return H


def monitor_constraints(gamma_rr, K_rr):
    """
    Monitor the Hamiltonian constraint and return its maximum value for stability analysis.
    """
    H = compute_hamiltonian_constraint(gamma_rr, K_rr)
    max_H = np.max(np.abs(H))
    return max_H


# Evolution Functions
def evolve_lapse_1pluslog(alpha, K, dt):
    return alpha - 2 * alpha * K * dt


def evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt):
    return gamma_rr - 2 * alpha * K_rr * dt


def evolve_curvature(alpha, h_rr, K_rr, dt, dr):
    return K_rr - alpha * np.gradient(h_rr, dr, axis=0) * dt


# Parameters and Initialization
Nr = 20
Ntheta = 10
r_min, r_max = 1.0, 20.0
dt = 0.01
T_max = 1.0

r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(0.0, np.pi, Ntheta)
gamma_rr = np.ones((Nr, Ntheta))  # Metric component
K_rr = np.random.rand(Nr, Ntheta) * 0.1  # Extrinsic curvature (initial)
h_rr = np.random.rand(Nr, Ntheta) * 0.1  # Perturbations
alpha = np.ones(Nr)  # Lapse function
beta_r = np.zeros(Nr)  # Shift vector

# Time Evolution Loop with Stability Analysis
time = 0.0
while time < T_max:
    # Evolve the lapse, metric, and curvature
    alpha = evolve_lapse_1pluslog(alpha, K_rr.mean(axis=1), dt)
    gamma_rr = evolve_metric(alpha[:, None], beta_r[:, None], K_rr, gamma_rr, dt)
    K_rr = evolve_curvature(alpha[:, None], h_rr, K_rr, dt, r[1] - r[0])

    # Monitor the Hamiltonian constraint
    max_H = monitor_constraints(gamma_rr, K_rr)
    print(f"Time: {time:.2f}, Max Hamiltonian Constraint: {max_H:.5e}")

    # Increment time
    time += dt

# Final Output for Verification
print("Final Gamma_rr:", gamma_rr)
print("Final K_rr:", K_rr)
