# Implementation of Higher-Order Numerical Methods (Runge-Kutta 4th Order)

def rk4_step(f, y, t, dt, *args):
    """
    Perform a single 4th-order Runge-Kutta step.
    Args:
        f: Function representing the derivative dy/dt = f(t, y).
        y: Current value of the variable.
        t: Current time.
        dt: Time step.
        *args: Additional arguments for the function f.
    Returns:
        Updated value of y after one RK4 step.
    """
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Example Function for Testing RK4: Simple Evolution of a Variable
def example_evolution(t, y, alpha, beta_r):
    """
    Example evolution function for gamma_rr in a simplified metric evolution.
    Args:
        t: Current time.
        y: Current value of the variable (gamma_rr).
        alpha: Lapse function (constant for testing).
        beta_r: Shift vector (constant for testing).
    Returns:
        Time derivative of gamma_rr.
    """
    return -2 * alpha * beta_r * y


# Constants and Initial Conditions
alpha = 1.0  # Lapse function (constant)
beta_r = -0.5  # Shift vector (constant)
gamma_rr_initial = 1.0  # Initial value of gamma_rr
t_initial = 0.0  # Initial time
t_final = 2.0  # Final time
dt = 0.1  # Time step

# Time Evolution using RK4
t_values = [t_initial]
gamma_rr_values = [gamma_rr_initial]
gamma_rr = gamma_rr_initial
t = t_initial

while t < t_final:
    gamma_rr = rk4_step(example_evolution, gamma_rr, t, dt, alpha, beta_r)
    t += dt
    t_values.append(t)
    gamma_rr_values.append(gamma_rr)

# Output Results for Verification
t_values, gamma_rr_values
