from operator import truediv
import setup_path

import airsim
import numpy as np
import random
import time
import math
from gym.spaces import Box

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

        self.car = airsim.CarClient()
        self.car.confirmConnection()

        self.car.enableApiControl(True, "A_Target")
        self.car.enableApiControl(True, "B_Adversarial")
        self.car.enableApiControl(True, "C_Front")
        

        self.target_car_controls = airsim.CarControls("A_Target")
        self.adversarial_car_controls = airsim.CarControls("B_Adversarial")
        self.front_car_controls = airsim.CarControls("C_Front")

        self.steering_scale = 0.15

    # for (get_state_of_target)
        self.state_size = 7

    # for (get_reward)
        self.distance_threshold = 20
        self.reward_scale_factor = 100
        self.distance_initial_value = 20
        self.dist_reward_scale = 0.1

        self.min_x = -10
        self.middle_x = 4
        self.max_x = 23

        self.min_y = -10
        self.middle_y = -7
        self.max_y = 5

        # self.X_F1 = REWARD_FUNC([( (self.min_x + self.middle_x)/2, 0) , (self.middle_x,1)])
        # self.X_F2 = REWARD_FUNC([( (self.max_x + self.middle_x)/2, 0) , (self.middle_x,1)])

        # self.Y_F1 = REWARD_FUNC([( (self.min_y + self.middle_y)/2, 0) , (self.middle_y,1)])
        # self.Y_F2 = REWARD_FUNC([( (self.max_y + self.middle_y)/2, 0) , (self.middle_y,1)])
    
        self.X_F1 = REWARD_FUNC([(-10, -1) , (4,1)])
        self.X_F2 = REWARD_FUNC([(23, -1) , (4,1)])
        
        self.Y_F1 = REWARD_FUNC([(-10, -1) , (-7,1)])
        self.Y_F2 = REWARD_FUNC([(5, -1) , (-7,1)])


        


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

        self.car.setCarControls(self.target_car_controls, "A_Target")



    def set_car_control_of_adversarial(self, throttle, steering, brake):
        self.adversarial_car_controls.throttle = throttle
        self.adversarial_car_controls.steering = steering
        self.adversarial_car_controls.brake = brake

        self.car.setCarControls(self.adversarial_car_controls, "B_Adversarial")



    def set_car_control_of_front(self, throttle, brake):
        self.front_car_controls.throttle = throttle
        self.front_car_controls.brake = brake
        self.car.setCarControls(self.front_car_controls, "C_Front")



    def set_car_control_of_front_random(self):
        front_throttle_value = self.front_car_controls.throttle

        change = (random.random() - 0.5) * 0.1
        front_throttle_value += change

        # 양 극에 도달했을 때의 처리를 변경
        if front_throttle_value > 0.90:
            front_throttle_value -= change  # 최대를 넘으면 변화량을 빼서 조정
        elif front_throttle_value < 0.30:
            front_throttle_value -= change  # 최소를 넘으면 변화량을 빼서 조정

        self.front_car_controls.throttle = front_throttle_value
        self.car.setCarControls(self.front_car_controls, "C_Front")

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
        if(self.min_x < distance_x < self.middle_x) : x_reward = self.X_F1.get_reward(adv_car_x)
        elif(self.middle_x <= distance_x < self.max_x) : x_reward = self.X_F2.get_reward(adv_car_x)
        else:
            x_reward = 0
            reward -= 50
            done = True
            print("ENDCODE : CONTROL_X01 ; "+str(distance_x))

        if(self.min_y <= adv_car_y <= self.middle_y) : y_reward = self.Y_F1.get_reward(adv_car_y)
        elif(self.middle_y < adv_car_y < self.max_y) : y_reward = self.Y_F2.get_reward(adv_car_y)
        else:
            y_reward = 0
            reward -= 50
            done = True
            print("ENDCODE : CONTROL_Y01 ; "+str(adv_car_y))

        #구간 내에서 브레이크 = 추가점수 ?

        reward += x_reward + y_reward

        collision_info_1 = self.car.simGetCollisionInfo("A_Target")
        collision_info_2 = self.car.simGetCollisionInfo("B_Adversarial")
        collision_info_3 = self.car.simGetCollisionInfo("C_Front")

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

        target_car_state = self.car.getCarState("A_Target")
        adversarial_car_state = self.car.getCarState("B_Adversarial")
        front_car_state = self.car.getCarState("C_Front")

        A_state = self.get_state_of_target(target_car_state)
        B_state = self.get_state_of_adversarial(adversarial_car_state)
        C_state = self.get_state_of_front(front_car_state)

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
        
        self.set_car_control_of_adversarial(throttle, steering, Is_brake)

#----------------------------------------------------------
#                7. methods for Ego agent
#----------------------------------------------------------

    def get_Ego_state(self):

        target_car_state = self.car.getCarState("A_Target")

        state = np.zeros(11)
        for i in range (9):
            sensor_name = "Distance_"        
            sensor_name = sensor_name + str(i)  
            state[i] = round(self.car.getDistanceSensorData(distance_sensor_name=sensor_name, vehicle_name="A_Target").distance, 3)

        state[9] = round(target_car_state.speed, 3)
        state[10] = target_car_state.gear

        state = np.reshape(state, [1, 11])

    # sate = [[ Laser sensor 9ea + target_speed + target_gear ]] <--numpy array
        return state.tolist()[0]

    def step_for_Ego(self, action):
        if( 0.1 <= action[1] ): Is_brake = 0
        else: Is_brake = 1
        
        self.set_car_control_of_target(float(action[0]), Is_brake)


if __name__ == "__main__":
    k = ENV()
    k.reset()


    k.set_car_control_of_adversarial(0.7, -0.15, 0)