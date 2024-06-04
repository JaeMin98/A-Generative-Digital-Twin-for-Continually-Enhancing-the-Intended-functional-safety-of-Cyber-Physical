import setup_path
import airsim
import numpy as np
import random
import time
from gym.spaces import Box

class ENV():
    def __init__(self):
    # define state and action space
        self.action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float)
        self.observation_space_size = 11

    # base parameter setting
        self._max_episode_steps = 8192 * 2

        self.car = airsim.CarClient()
        self.car.confirmConnection()

        self.car.enableApiControl(True, "A_Target")
        self.car.enableApiControl(True, "C_Front")

        self.target_car_controls = airsim.CarControls("A_Target")
        self.front_car_controls = airsim.CarControls("C_Front")

    # for (get_state_of_target)
        self.state_size = 11

    # for (get_reward)
        self.distance_threshold = 20
        self.reward_scale_factor = 100
        self.distance_initial_value = 20
        self.dist_reward_scale = 0.1

    def reset(self):
        self.car.reset()
        self.set_car_control_of_target(0.65, 0)
        self.set_car_control_of_front(0.65, 0)
        time.sleep(2)

        state, _, __, ___ = self.observation()

        return state
    
    def get_state_of_target(self, target_car_state):
        state = np.zeros(self.state_size)
        for i in range (9):
            sensor_name = "Distance_"        
            sensor_name = sensor_name + str(i)  
            state[i] = round(self.car.getDistanceSensorData(distance_sensor_name=sensor_name, vehicle_name="A_Target").distance, 3)

        state[9] = round(target_car_state.speed, 3)
        state[10] = target_car_state.gear
        state = np.reshape(state, [1, self.state_size])
    # sate = [[ Laser sensor 9ea + target_speed + target_gear ]] <--numpy array
        return state.tolist()[0]


    def set_car_control_of_target(self, throttle, brake):
        if throttle > 0.95: throttle = 0.95
        elif throttle < 0.35: throttle = 0.35

    # 0.0 < throttle < 1.0
        self.target_car_controls.throttle = throttle
    # brake <-- True or False
        self.target_car_controls.brake = brake

        self.car.setCarControls(self.target_car_controls, "A_Target")

    def set_car_control_of_front(self, throttle, brake):
    # 0.0 < throttle < 1.0
        self.front_car_controls.throttle = throttle
    # brake <-- True or False
        self.front_car_controls.brake = brake
        self.car.setCarControls(self.front_car_controls, "C_Front")

    def set_car_control_of_front_random(self):
        front_throttle_value = 0.65
        front_throttle_value += (random.randrange(0, 10) - 5) * 0.01

        if front_throttle_value > 0.95:
            front_throttle_value = 0.95
        elif front_throttle_value < 0.35:
            front_throttle_value = 0.35
    #?????????????????????????????????????????????????????????????? what about C_Front car's brake?
        self.front_car_controls.throttle = front_throttle_value
        self.car.setCarControls(self.front_car_controls, "C_Front")


    def get_reward(self, target_car_state, front_car_state):
        target_car_x = target_car_state.kinematics_estimated.position.x_val
        front_car_x = front_car_state.kinematics_estimated.position.x_val
        
        distance = front_car_x - target_car_x + self.distance_initial_value
        distance_gap = float(abs(distance - self.distance_threshold))
        dist_reward = (0.025) * self.dist_reward_scale * (distance_gap-20.0)**2  #(0,10)

        reward = dist_reward
        done = False
        success = False

        if distance_gap >= self.distance_threshold -2:
            done = True
            reward -= 300

        if target_car_x >= 300:
            done = True
            success = True
            reward += 300

        return reward,done,success

    def observation(self):

        target_car_state = self.car.getCarState("A_Target")
        front_car_state = self.car.getCarState("C_Front")

        state = self.get_state_of_target(target_car_state)
        reward, done, success = self.get_reward(target_car_state, front_car_state)

        return state, reward, done, success

    def step(self, action):
        if(action[1] <= 0.9): Is_brake = 0
        else: Is_brake = 0
        
        self.set_car_control_of_target(float(action[0]), Is_brake)



if __name__ == "__main__":
    k = ENV()
    k.reset()

    # k.set_car_control_of_target(0.99, 0)
    # k.set_car_control_of_front_random()
    # time.sleep(2)

    # total = 0
    # while(1):
    #     state, reward, done = k.observation()
        
    #     total += reward

    #     time.sleep(0.01)

    #     if done :
    #         print(total)
    #         total = 0

    #         k.reset()

    #         k.set_car_control_of_target(0.99, 0)
    #         k.set_car_control_of_front_random()
    #         time.sleep(2)

        