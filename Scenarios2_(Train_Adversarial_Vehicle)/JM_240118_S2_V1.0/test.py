import numpy as np
import matplotlib.pyplot as plt

points = [(-7, 1), (-6, 0), (3, -1)]
x_points = [p[0] for p in points]
y_points = [p[1] for p in points]

coefficients = np.polyfit(x_points, y_points, 2)
a, b, c = coefficients

x = np.linspace(-8, 4, 400)
y = a * x**2 + b * x + c

plt.plot(x, y, label=f'y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}')
plt.scatter(x_points, y_points, color='red')
plt.legend()
plt.show()
