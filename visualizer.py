
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self):

        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(7, 7))

    def plot(self, particles, t):

        self.ax.clear()

        x = np.array([p.x for p in particles])
        y = np.array([p.y for p in particles])

        infected = np.array([
            p.infected for p in particles
        ])

        masked = np.array([
            p.mask_on for p in particles
        ])

        # All particles
        self.ax.plot(x, y, 'k.', markersize=3)

        # Infected particles
        self.ax.plot(
            x[infected],
            y[infected],
            'ro',
            markersize=5
        )

        # Masked particles
        self.ax.plot(
            x[masked],
            y[masked],
            'g*',
            markersize=6
        )

        self.ax.grid(True)

        self.ax.set_title(f"Time = {t:.3f}")

        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)

        self.ax.set_aspect('equal')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def finish(self):

        plt.ioff()
        plt.show()
