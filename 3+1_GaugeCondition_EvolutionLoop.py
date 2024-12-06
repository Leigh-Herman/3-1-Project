import numpy as np

# Gauge condition evolution
def evolve_lapse_1pluslog(alpha, K, dt):
    return alpha - 2 * alpha * K * dt

def evolve_shift_gammadriver(beta, B, Gamma, eta, dt):
    B_new = B + dt * (Gamma - eta * B)
    beta_new = beta + (3 / 4) * B_new * dt
    return beta_new, B_new

# Metric and curvature evolution
def evolve_metric(alpha, beta_r, K_rr, gamma_rr, dt):
    return gamma_rr - 2 * alpha * K_rr * dt

def evolve_curvature(alpha, h_rr, K_rr, dt, dr):
    return K_rr - alpha * np.gradient(h_rr, dr, axis=0) * dt

# Boundary conditions
def apply_outgoing_wave_boundary(h, dr, dt):
    h[0, :] = h[1, :]  # Inner edge
    h[-1, :] = h[-2, :] - (dt / dr) * (h[-2, :] - h[-3, :])  # Outer edge
    return h

def apply_inner_boundary_condition(h):
    h[0, :] = 0  # Freeze perturbation at the inner boundary
    return h

# Parameters and grid setup
Nr = 100
Ntheta = 50
r_min, r_max = 1.0, 20.0
theta_min, theta_max = 0.0, np.pi
dr = (r_max - r_min) / Nr
dt = 0.01
T_max = 1.0

r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)

# Initialization
gamma_rr = np.ones((Nr, Ntheta))
K_rr = np.zeros((Nr, Ntheta))
h_rr = np.random.rand(Nr, Ntheta) * 0.1
alpha = np.ones(Nr)
beta = np.zeros(Nr)
B = np.zeros(Nr)
Gamma = np.random.rand(Nr) * 0.01
eta = 2.0

# Evolution loop
time = 0.0
while time < T_max:
    alpha = evolve_lapse_1pluslog(alpha, K_rr.mean(axis=1), dt)
    beta, B = evolve_shift_gammadriver(beta, B, Gamma, eta, dt)
    gamma_rr = evolve_metric(alpha[:, None], beta[:, None], K_rr, gamma_rr, dt)
    K_rr = evolve_curvature(alpha[:, None], h_rr, K_rr, dt, dr)
    h_rr = apply_outgoing_wave_boundary(h_rr, dr, dt)
    h_rr = apply_inner_boundary_condition(h_rr)
    H = (K_rr ** 2) / gamma_rr
    if time % 0.1 < dt:
        print(f"Time: {time:.2f}, Max Hamiltonian Constraint: {np.max(H):.5f}")
    time += dt

# Final Output
print("Final Lapse Alpha:", alpha)
print("Final Shift Vector Beta:", beta)
print("Max Hamiltonian Constraint:", np.max(H))
