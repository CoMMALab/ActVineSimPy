import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")

point, = ax.plot([], [], "o", ms=10, color="tab:blue")  # empty at first

def update(i):
    theta = i * 0.1
    x, y = np.cos(theta), np.sin(theta)
    point.set_data([x], [y])
    ax.set_title(f"frame {i}")
    return [point]

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()