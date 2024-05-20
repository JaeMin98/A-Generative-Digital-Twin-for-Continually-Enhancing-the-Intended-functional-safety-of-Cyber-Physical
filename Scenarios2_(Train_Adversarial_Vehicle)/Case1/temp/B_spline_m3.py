import numpy as np
import matplotlib.pyplot as plt


class BSpline:
    def __init__(self, control_points):
        # 컨트롤 포인트 초기화
        self.control_points = np.array(control_points)
        self.degree = 3  # 3차 B-스플라인
        self.n = len(control_points) - 1
        self.knots = np.array([0, 0, 0, 0] + list(range(1, self.n - 2)) + [self.n - 2, self.n - 2, self.n - 2, self.n - 2])

    def basis_function(self, i, k, t):
        # B-스플라인 베이시스 함수 계산
        if k == 0:
            return 1.0 if self.knots[i] <= t < self.knots[i+1] else 0.0
        else:
            d1 = self.knots[i+k] - self.knots[i]
            d2 = self.knots[i+k+1] - self.knots[i+1]
            N1 = 0 if d1 == 0 else (t - self.knots[i]) / d1 * self.basis_function(i, k-1, t)
            N2 = 0 if d2 == 0 else (self.knots[i+k+1] - t) / d2 * self.basis_function(i+1, k-1, t)
            return N1 + N2

    def compute_point(self, t):
        # 주어진 t 값에 대해 B-스플라인 곡선의 포인트 계산
        point = np.zeros(2)
        for i in range(self.n + 1):
            b = self.basis_function(i, self.degree, t)
            point += b * self.control_points[i]
        return point

    def compute_curve(self, num_points=256):
        # B-스플라인 곡선의 포인트를 계산하여 반환
        points = []
        for t in np.linspace(self.knots[self.degree], self.knots[-self.degree-1], num_points):
            points.append(self.compute_point(t))
        return np.array(points)

class BSplineVisualizer(BSpline):
    def plot(self, num_points=100):
        # B-스플라인 곡선과 컨트롤 포인트를 시각화
        curve = self.compute_curve(num_points=num_points)  # 'num_dict' 대신 'num_points'를 사용
        curve[-1] = curve[-2]
        control_points = np.array(self.control_points)
        print(curve)
        print(control_points)

        plt.figure(figsize=(8, 5))
        plt.plot(curve[:, 0], curve[:, 1], label='B-Spline Curve')
        plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')
        plt.title('B-Spline Curve Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()

# 사용 예시
control_points = [(0, -3), (1, 5), (2, 8), (4, -5), (5, 10)]
spline_visualizer = BSplineVisualizer(control_points)
spline_visualizer.plot()