from operator import truediv
import airsim
import numpy as np
import time
from gym.spaces import Box
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
    def plot(self, num_points=256):
        # B-스플라인 곡선과 컨트롤 포인트를 시각화
        curve = self.compute_curve(num_points=num_points)  # 'num_dict' 대신 'num_points'를 사용
        curve[-1] = curve[-2]
        control_points = np.array(self.control_points)

        plt.figure(figsize=(8, 5))
        plt.plot(curve[:, 0], curve[:, 1], label='B-Spline Curve')
        plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')
        plt.title('B-Spline Curve Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid(True)
        plt.show()

class ENV():
#----------------------------------------------------------
#              1. initialization parameters
#----------------------------------------------------------
    def __init__(self):
    # define state and action space (전진, 회전, 브레이크)
        self.action_space = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space_size = 7
        self.action_space_of_ego = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space_size_of_ego = 11
    # base parameter setting
        self._max_episode_steps = 8192 * 2
        try:
            self.car = airsim.CarClient()
            self.car.confirmConnection()
            self.car.enableApiControl(True, "A_Target")
            self.car.enableApiControl(True, "B_Adversarial")
            self.car.enableApiControl(True, "C_Front")
            self.target_car_controls = airsim.CarControls("A_Target")
            self.adversarial_car_controls = airsim.CarControls("B_Adversarial")
            self.front_car_controls = airsim.CarControls("C_Front")
        except:
            print("AIRSIM ERROR_01 : request failed")
        self.steering_scale = 0.15
    # for (get_state_of_target)
        self.state_size = 7
    # for (get_reward)
        self.distance_threshold = 20
        self.reward_scale_factor = 100
        self.distance_initial_value = 20
        self.dist_reward_scale = 0.1
#----------------------------------------------------------
#                     2. reset
#----------------------------------------------------------
    def reset(self):
        self.car.reset()
        self.set_car_control_of_target(0.6, 0)
        self.set_car_control_of_adversarial(0.6, 0, 0)
        self.set_car_control_of_front(0.6, 0)
        time.sleep(2)
        # state, _, __, ___ = self.observation()
        # return state
#----------------------------------------------------------
#                   3. get state
#----------------------------------------------------------
    def get_state_of_front(self, front_car_state):
        current_state_of_front_car = []
        current_state_of_front_car.append(  round(front_car_state.kinematics_estimated.position.x_val,3)  )
        current_state_of_front_car.append(  round(front_car_state.speed, 3)  )
        return current_state_of_front_car
    def get_state_of_target(self, target_car_state):
        current_state_of_target_car = []
        current_state_of_target_car.append(  round(target_car_state.kinematics_estimated.position.x_val,3)  )
        current_state_of_target_car.append(  round(target_car_state.speed, 3)  )
        return current_state_of_target_car
    def get_state_of_adversarial(self, adversarial_car_state):
        current_state_of_adversarial_car = []
        current_state_of_adversarial_car.append(  round(adversarial_car_state.kinematics_estimated.position.x_val,3)  )
        current_state_of_adversarial_car.append(  round(adversarial_car_state.kinematics_estimated.position.y_val,3)  )
        current_state_of_adversarial_car.append(  round(adversarial_car_state.speed, 3)  )
        return current_state_of_adversarial_car
#----------------------------------------------------------
#                  4. control car
#----------------------------------------------------------
    def set_car_control_of_target(self, throttle, brake):
        if throttle > 0.95: throttle = 0.95
        elif throttle < 0.25: throttle = 0.25
        self.target_car_controls.throttle = throttle
        self.target_car_controls.brake = brake

        try:
            self.car.setCarControls(self.target_car_controls, "A_Target")
        except:
            print("AIRSIM ERROR_02 : request failed")

    def set_car_control_of_adversarial(self, throttle, steering, brake):
        self.adversarial_car_controls.throttle = throttle
        self.adversarial_car_controls.steering = steering
        self.adversarial_car_controls.brake = brake
        try:
            self.car.setCarControls(self.adversarial_car_controls, "B_Adversarial")
        except:
            print("AIRSIM ERROR_03 : request failed")

    def set_car_control_of_front(self, throttle, brake):
        self.front_car_controls.throttle = throttle
        self.front_car_controls.brake = brake
        try:
            self.car.setCarControls(self.front_car_controls, "C_Front")
        except:
            print("AIRSIM ERROR_04 : request failed")

if __name__ == "__main__":

    # 사용 예시
    control_points = [(0, -3), (1, 5), (2, 8), (4, -5), (5, 10)]
    spline_visualizer = BSplineVisualizer(control_points)
    print(spline_visualizer.compute_curve())
    spline_visualizer.plot()

    k = ENV()
    k.reset()
    k.set_car_control_of_adversarial(0.6, 0, 0)