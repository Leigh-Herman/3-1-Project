import numpy as np
from scipy.special import sph_harm

def initialize_perturbation_spherical(r, theta, l, m, amplitude):
    """
    Initialize the perturbation using spherical harmonics.
    Args:
        r: Radial grid points.
        theta: Angular grid points.
        l: Degree of the spherical harmonic.
        m: Order of the spherical harmonic.
        amplitude: Amplitude of the perturbation.
    Returns:
        Initialized perturbation array h(r, theta).
    """
    # Compute the spherical harmonic function Y_lm(theta, phi) (real part only for simplicity)
    Y_lm = sph_harm(m, l, 0, theta).real  # Setting phi=0 for 2D axisymmetry
    perturbation = amplitude * np.outer(r, Y_lm)
    return perturbation

# Example Parameters
Nr = 50  # Radial points
Ntheta = 25  # Angular points
r_min, r_max = 1.0, 20.0
theta_min, theta_max = 0.0, np.pi
l = 2  # Degree of spherical harmonic
m = 0  # Order of spherical harmonic
amplitude = 0.1  # Amplitude of the perturbation

# Create Grid
r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)

# Initialize Perturbation
h_perturbation = initialize_perturbation_spherical(r, theta, l, m, amplitude)

# Output for Verification
print("Perturbation Shape:", h_perturbation.shape)
print("Perturbation (Sample):", h_perturbation)
