import matplotlib.pyplot as plt
import numpy as np

def victor():
    def func(x, y):
        val = 2.8*x**2 * (x**2*(2.5*x**2+y**2-2)+1.2*y**2 * (y*(3*y-0.75)-6.0311)+3.09)+0.98*y**2 * ((y**2-3.01)*y**2+3)-1.005
        return abs(val) < 0.02
    n = 3000
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    grid = np.zeros((n, n))
    y, x = np.meshgrid(x, y)
    z = func(x, y)
    z = z[:-1, :-1]
    z_min, z_max = 0, 1
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, z, vmin=z_min, vmax=z_max)
    ax.set_title('Viktooooor')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()

victor()