from joblib import Parallel, delayed
import numpy as np


def compute_hamiltonian_constraint(gamma_rr, K_rr):
    """
    Compute the Hamiltonian constraint:
    H = K_rr^2 / gamma_rr.
    """
    H = np.nan_to_num(K_rr ** 2 / gamma_rr)  # Avoid division by zero with nan_to_num
    return H


def monitor_constraints_parallel(gamma_rr, K_rr, n_jobs=-1):
    """
    Monitor the Hamiltonian constraint using parallelism.
    Args:
        gamma_rr: Metric component (radial-radial).
        K_rr: Extrinsic curvature component (radial-radial).
        n_jobs: Number of parallel jobs (-1 uses all available cores).
    Returns:
        Maximum Hamiltonian constraint across all grid points.
    """

    def process_chunk(gamma_chunk, K_chunk):
        H = compute_hamiltonian_constraint(gamma_chunk, K_chunk)
        return np.max(np.abs(H))

    chunk_results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(gamma_rr[i, :], K_rr[i, :]) for i in range(gamma_rr.shape[0])
    )
    return max(chunk_results)


# Parameters for Testing
Nr = 20
Ntheta = 10
gamma_rr = np.ones((Nr, Ntheta))  # Metric component
K_rr = np.random.rand(Nr, Ntheta) * 0.1  # Extrinsic curvature

# Example Parallel Stability Monitoring
max_H = monitor_constraints_parallel(gamma_rr, K_rr)
print(f"Max Hamiltonian Constraint (Parallel): {max_H:.5e}")
