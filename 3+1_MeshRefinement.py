import numpy as np

# Adaptive Mesh Refinement (AMR) Function
def refine_mesh(h, r, threshold):
    """
    Refine the mesh by adding additional points in regions where the gradient of h exceeds a threshold.
    Args:
        h: Perturbation array (Nr x Ntheta).
        r: Radial grid points.
        threshold: Refinement threshold based on gradient magnitude.
    Returns:
        Refined h and r arrays with additional points where needed.
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

# Example Simulation with Refinement
Nr = 20  # Initial radial points
Ntheta = 10  # Angular points
r = np.linspace(1.0, 20.0, Nr)  # Initial radial grid
theta = np.linspace(0.0, np.pi, Ntheta)

# Initial perturbation
h = np.sin(np.outer(r, np.ones(Ntheta))) * np.cos(theta)  # Example function

# Refinement threshold
threshold = 0.2

# Perform refinement
refined_r, refined_h = refine_mesh(h, r, threshold)

# Output for Verification
print("Refined radial grid:", refined_r)
print("Refined perturbation array shape:", refined_h.shape)
