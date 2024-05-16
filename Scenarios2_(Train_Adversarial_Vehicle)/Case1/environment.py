from operator import truediv
import setup_path
import airsim
import numpy as np
import random
import time
import math
from gym.spaces import Box
class ENV():
#----------------------------------------------------------
#              1. initialization parameters
#----------------------------------------------------------
    def __init__(self):
    # define state and action space (전진, 회전, 브레이크)
        self.action_space = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float)
        self.observation_space_size = 7
        self.action_space_of_ego = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float)
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
        state, _, __, ___ = self.observation()
        return state
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

#----------------------------------------------------------
#                 5. reward function
#----------------------------------------------------------
    def get_reward(self, adversarial_car_state, target_car_state):
        target_car_x = target_car_state.kinematics_estimated.position.x_val
        target_car_y = target_car_state.kinematics_estimated.position.y_val
        adv_car_x = adversarial_car_state.kinematics_estimated.position.x_val
        adv_car_y = adversarial_car_state.kinematics_estimated.position.y_val
        reward = 0
        done = False
        success = False
        distance_x = adv_car_x - target_car_x
        if(-10 < distance_x < 6) : x_reward = 0.00024*(distance_x+10)**3
        elif(6 <= distance_x < 23) : x_reward = -0.0002*(distance_x-23)**3
        else:
            x_reward = 0
            reward -= 50
            done = True
            print("ENDCODE : CONTROL_X01 ; "+str(distance_x))
        if(-10 <= adv_car_y <= -7) : y_reward = 0.037*(adv_car_y+10)**3
        elif(-7 < adv_car_y < 5) : y_reward = -0.00058*(adv_car_y-5)**3
        else:
            y_reward = 0
            reward -= 50
            done = True
            print("ENDCODE : CONTROL_Y01 ; "+str(adv_car_y))
        #구간 내에서 브레이크 = 추가점수 ?
        reward += x_reward + y_reward
        try:
            collision_info_1 = self.car.simGetCollisionInfo("A_Target")
            collision_info_2 = self.car.simGetCollisionInfo("B_Adversarial")
            collision_info_3 = self.car.simGetCollisionInfo("C_Front")
        except:
            collision_info_1 = False
            collision_info_2 = False
            collision_info_3 = False
            print("AIRSIM ERROR_06 : request failed")
        if (collision_info_3.has_collided == True) and (collision_info_2.has_collided == True):
            done = True
            reward -= 100
            print("ENDCODE : COLLISION_01")
        if (collision_info_1.has_collided == True) and (collision_info_2.has_collided == True):
            # not ROI collision
            if adv_car_x - target_car_x < 4:
                reward -= 50
                done = True
                print("ENDCODE : COLLISION_02")
            else: # ROI for collision
                reward += 200
                done = True
                success = True
                print("ENDCODE : COLLISION_03")
        return reward,done,success
#----------------------------------------------------------
#         6. observation & step for learning
#----------------------------------------------------------
    def observation(self):
        while(True):
            try:
                target_car_state = self.car.getCarState("A_Target")
                adversarial_car_state = self.car.getCarState("B_Adversarial")
                front_car_state = self.car.getCarState("C_Front")
                A_state = self.get_state_of_target(target_car_state)
                B_state = self.get_state_of_adversarial(adversarial_car_state)
                C_state = self.get_state_of_front(front_car_state)
                break
            except:
                print("AIRSIM ERROR_07 : request failed")
        state = A_state + B_state + C_state
        reward, done, success = self.get_reward(adversarial_car_state, target_car_state)
        return state, reward, done, success
    
    def step(self, action):
        # 쓰로틀 조정
        throttle = float(action[0])
        if throttle > 0.95: throttle = 0.95
        elif throttle < 0.25: throttle = 0.25
        # 스티어링 조정
        steering = float(action[1]) #  steering : [0 ~ 1.0]
        steering = steering - 0.5 # steering : [-0.5 ~ 0.5]
        steering = steering * 2 # steering : [-1.0 ~ 1.0]
        steering = steering * self.steering_scale  # -> steering : [-0.15 ~ 0.15]
        if steering > 0.15 : steering = 0.15
        elif steering < -0.15 : steering = -0.15
        # 브레이크 조정
        if( 0.1 <= action[2] ): Is_brake = 0
        else: Is_brake = 1
        while(True):
            try:
                self.set_car_control_of_adversarial(throttle, steering, Is_brake)
                break
            except:
                print("AIRSIM ERROR_08 : request failed")

if __name__ == "__main__":
    k = ENV()
    k.reset()
    k.set_car_control_of_adversarial(0.7, -0.15, 0)