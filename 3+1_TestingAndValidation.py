import numpy as np

def validate_hamiltonian_constraint(H_data, tolerance=1e-3):
    """
    Validate the Hamiltonian constraint over time.
    Args:
        H_data: 3D array of Hamiltonian constraint values (time_steps x Nr x Ntheta).
        tolerance: Acceptable upper limit for the maximum Hamiltonian constraint.
    Returns:
        Validation result (True/False) and the time indices where violations occur.
    """
    time_steps = H_data.shape[0]
    violations = []

    for t in range(time_steps):
        max_H = np.max(np.abs(H_data[t, :, :]))
        if max_H > tolerance:
            violations.append((t, max_H))

    return len(violations) == 0, violations

def test_energy_conservation(gamma_rr, initial_energy, tolerance=1e-3):
    """
    Test energy conservation by comparing current energy to the initial energy.
    Args:
        gamma_rr: Metric component at the current timestep.
        initial_energy: Energy calculated at the initial timestep.
        tolerance: Acceptable relative deviation from initial energy.
    Returns:
        Validation result (True/False) and current energy.
    """
    current_energy = np.sum(gamma_rr)  # Simplified energy metric for testing
    deviation = np.abs(current_energy - initial_energy) / initial_energy
    return deviation <= tolerance, current_energy

# Example Data for Testing
Nr = 50
Ntheta = 25
T_max = 2.0
dt = 0.1
time_steps = int(T_max / dt)

# Simulated Hamiltonian constraint data (synthetic for testing)
H_data = np.random.rand(time_steps, Nr, Ntheta) * 0.001  # Within tolerance

# Simulated metric data for energy conservation test
gamma_rr_initial = np.ones((Nr, Ntheta))
gamma_rr_final = gamma_rr_initial + np.random.rand(Nr, Ntheta) * 0.01  # Slight variation
initial_energy = np.sum(gamma_rr_initial)

# Validate Hamiltonian Constraint
valid, violations = validate_hamiltonian_constraint(H_data, tolerance=1e-3)
print("Hamiltonian Constraint Validation Passed:", valid)
if not valid:
    print("Violations Found at Time Indices (Frame, Max H):", violations)

# Test Energy Conservation
energy_valid, current_energy = test_energy_conservation(gamma_rr_final, initial_energy, tolerance=1e-3)
print("Energy Conservation Test Passed:", energy_valid)
print("Initial Energy:", initial_energy, "Current Energy:", current_energy)
