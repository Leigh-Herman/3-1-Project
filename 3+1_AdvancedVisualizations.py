import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example Data for Visualization
Nr = 50  # Radial points
Ntheta = 25  # Angular points
T_max = 2.0  # Maximum simulation time
dt = 0.1  # Time step

r = np.linspace(1.0, 20.0, Nr)
theta = np.linspace(0.0, np.pi, Ntheta)
time_steps = int(T_max / dt)

# Generate synthetic Hamiltonian constraint data for visualization
H_data = np.random.rand(time_steps, Nr, Ntheta) * 0.01

# Create 2D Cartesian grid for visualization
r_mesh, theta_mesh = np.meshgrid(r, theta)
x = r_mesh * np.sin(theta_mesh)
y = r_mesh * np.cos(theta_mesh)

# Visualization Setup
fig, ax = plt.subplots(figsize=(8, 6))
cbar = None

def update(frame):
    global cbar
    ax.clear()
    H = H_data[frame, :, :]
    contour = ax.contourf(x, y, H.T, levels=100, cmap="viridis")
    if cbar is None:
        cbar = plt.colorbar(contour, ax=ax, label="Hamiltonian Constraint")
    ax.set_title(f"Hamiltonian Constraint at Time {frame * dt:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

# Create Animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=100)
ani.save("hamiltonian_constraint_evolution.gif", writer="pillow")
plt.close(fig)
