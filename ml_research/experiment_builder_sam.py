import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the loss function
def loss(x, y):
    return (x**2 + y**2)/20 + np.sin(5*x) + np.sin(5*y)

# Create a mesh grid of x and y values
X, Y = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1))

# Calculate the loss for each point in the mesh grid
Z = loss(X, Y)

# Plot the loss landscape in 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Loss')
plt.show()