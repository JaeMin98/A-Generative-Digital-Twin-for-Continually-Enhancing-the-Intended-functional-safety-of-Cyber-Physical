import numpy as np
import matplotlib.pyplot as plt



class REWARD_FUNC():
    def __init__(self, points):
        self.points = points
        self.make_reward_finction()

    def make_reward_finction(self):
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]

        coefficients = np.polyfit(x, y, 1)
        self.a, self.b = coefficients

    def get_reward(self, x):
        return self.a * (x) + self.b

    def get_graph(self):
        x_points = [p[0] for p in self.points]
        y_points = [p[1] for p in self.points]

        x = np.linspace(-20, 20, 400)
        y = self.a * x + self.b

        plt.plot(x, y, label=f'y = {self.a:.2f}x + {self.b:.2f}')
        plt.scatter(x_points, y_points, color='red')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    k = REWARD_FUNC([(23, -1) , (7,1)])
    k.get_graph()
    print(k.get_reward(0))