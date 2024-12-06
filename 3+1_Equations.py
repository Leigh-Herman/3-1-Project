from sympy import symbols, Function, diff, simplify, Matrix
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead

# Define symbolic variables and functions
t, r, theta, phi, epsilon = symbols('t r theta phi epsilon')
M = symbols('M', positive=True)  # Mass parameter for Schwarzschild spacetime

# Metric components in Painleve-Gullstrand coordinates
alpha_0 = Function('alpha_0')(t, r, theta, phi)  # Background lapse
beta_r = -Function('beta_r')(r, theta, phi)  # Background shift
gamma_rr = Function('gamma_rr')(r)
gamma_tt = -alpha_0**2
gamma_theta_theta = r**2
gamma_phi_phi = r**2 * Function('sin')(theta)**2

# Metric perturbations
h_tt = Function('h_tt')(t, r, theta, phi)
h_tr = Function('h_tr')(t, r, theta, phi)
h_rr = Function('h_rr')(t, r, theta, phi)
h_theta_theta = Function('h_theta_theta')(t, r, theta, phi)
h_phi_phi = Function('h_phi_phi')(t, r, theta, phi)

# Extrinsic curvature perturbations
K_rr = Function('K_rr')(t, r, theta, phi)
K_theta_theta = Function('K_theta_theta')(t, r, theta, phi)
K_phi_phi = Function('K_phi_phi')(t, r, theta, phi)

# Hamiltonian constraint: R - K_ij K^ij + K^2 = 0
Kij = Matrix([[K_rr, 0, 0], [0, K_theta_theta, 0], [0, 0, K_phi_phi]])
gamma_inv = Matrix([[1/gamma_rr, 0, 0], [0, 1/gamma_theta_theta, 0], [0, 0, 1/gamma_phi_phi]])
tr_K2 = simplify((Kij.T * gamma_inv * Kij).trace())
hamiltonian_constraint = simplify(tr_K2)

# Evolution equations
evol_metric = diff(gamma_rr, t) + 2 * alpha_0 * K_rr
evol_curvature = diff(K_rr, t) - alpha_0 * diff(h_rr, r)

# Collect results
results = {
    'Hamiltonian Constraint': hamiltonian_constraint,
    'Metric Evolution': evol_metric,
    'Curvature Evolution': evol_curvature,
}

results
print(results)