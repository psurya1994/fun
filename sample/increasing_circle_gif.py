import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')

rads = np.linspace(0, 2 * np.pi, 100)
ims = []
for i in range(20):
    ims.append(ax.plot(i*np.cos(rads), i*np.sin(rads)))

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=500)

ani.save("demo2.gif", writer='imagemagick')

plt.show()