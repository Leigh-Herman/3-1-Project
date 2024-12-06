import numpy as np

def compute_hamiltonian_constraint(gamma_rr, K_rr):
    """
    Compute the Hamiltonian constraint:
    H = R - K_ij K^ij + K^2
    For simplicity, this implementation uses:
    H = K_rr^2 / gamma_rr
    Args:
        gamma_rr: Metric component (radial-radial).
        K_rr: Extrinsic curvature component (radial-radial).
    Returns:
        Hamiltonian constraint array.
    """
    H = np.nan_to_num(K_rr**2 / gamma_rr)  # Avoid division by zero with nan_to_num
    return H

def monitor_constraints(gamma_rr, K_rr):
    """
    Monitor the Hamiltonian constraint and return its maximum value for stability analysis.
    """
    H = compute_hamiltonian_constraint(gamma_rr, K_rr)
    max_H = np.max(np.abs(H))
    return max_H

# Example Stability Analysis in Evolution Loop
Nr = 20
Ntheta = 10
gamma_rr = np.ones((Nr, Ntheta))  # Initialize gamma_rr
K_rr = np.random.rand(Nr, Ntheta) * 0.1  # Initialize small random values for K_rr
dt = 0.01  # Time step
T_max = 1.0  # Simulation time
time = 0.0

# Stability Monitoring
while time < T_max:
    # Simulate updates for gamma_rr and K_rr (simplified here as random evolution for testing)
    gamma_rr += np.random.rand(Nr, Ntheta) * 0.01
    K_rr += np.random.rand(Nr, Ntheta) * 0.01

    # Monitor constraints
    max_H = monitor_constraints(gamma_rr, K_rr)
    print(f"Time: {time:.2f}, Max Hamiltonian Constraint: {max_H:.5e}")

    # Increment time
    time += dt
