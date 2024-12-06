# Implementation of Boundary Conditions

def apply_outgoing_wave_boundary(h, dr, dt):
    """
    Apply outgoing wave boundary condition at the edges of the grid.
    Args:
        h: Perturbation array.
        dr: Radial step size.
        dt: Time step.
    Returns:
        Updated perturbation array with boundary condition applied.
    """
    # Outgoing wave condition: ∂h/∂t + ∂h/∂r = 0
    h[0, :] = h[1, :]  # Nearest to the center
    h[-1, :] = h[-2, :] - (dt / dr) * (h[-2, :] - h[-3, :])  # At the outer edge
    return h


def apply_inner_boundary_condition(h):
    """
    Apply inner boundary condition near the black hole horizon.
    Args:
        h: Perturbation array.
    Returns:
        Updated perturbation array with inner boundary condition applied.
    """
    # Freeze or smoothly damp perturbations near the inner boundary
    h[0, :] = 0  # Example: Freezing at the inner boundary
    return h


# Test Example
import numpy as np

# Parameters
Nr = 100  # Radial points
Ntheta = 50  # Angular points
dr = 0.1  # Radial step size
dt = 0.01  # Time step

# Initial perturbation
h = np.random.rand(Nr, Ntheta) * 0.1  # Small random perturbations for testing

# Apply boundary conditions
h = apply_outgoing_wave_boundary(h, dr, dt)
h = apply_inner_boundary_condition(h)

# Output for verification
h_boundary_applied = h.copy()
h_boundary_applied
