from operator import truediv
import setup_path
import airsim
import numpy as np
import random
import time
import math
import json
from gym.spaces import Box




class ENV():
#----------------------------------------------------------
#              1. initialization parameters
#----------------------------------------------------------
    def __init__(self):
    # define state and action space (전진, 회전, 브레이크)
        low = np.array([-3.0, -3.0, -1.0, -1.0, -1.0, 0.0])
        high = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0])
        self.action_space = Box(low=low, high=high,shape=(6,), dtype=np.float32)
        self.observation_space_size = 4
        self.num_of_steering = 3

        self.action_space_of_ego = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space_size_of_ego = 11

    # base parameter setting
        self._max_episode_steps = 256

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


        self.json_file_path = "/home/smartcps/Documents/AirSim/settings.json"

#----------------------------------------------------------
#                     2. reset
#----------------------------------------------------------
    def reset(self, x,y):
        self.collision_info_1 = False
        self.collision_info_2 = False

        self.car.reset()
        self.set_position(x,y)

        self.set_car_control_of_target(0.6, 0)
        self.set_car_control_of_adversarial(0.6, 0, 0)
        self.set_car_control_of_front(0.6, 0)
        time.sleep(2)
        state, _, __, ___ = self.observation()
        return state
    
    def set_position(self,x,y):
        position = airsim.Vector3r(x, y, -3)
        orientation = airsim.Quaternionr(0, 0, 0, 1)
        pose = airsim.Pose(position, orientation)

        self.car.simSetObjectPose("B_Adversarial", pose, True)

        position = airsim.Vector3r(-100, 0, -3)
        orientation = airsim.Quaternionr(0, 0, 0, 1)
        pose = airsim.Pose(position, orientation)

        self.car.simSetObjectPose("C_Front", pose, True)

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
        current_state_of_target_car.append(  round(target_car_state.kinematics_estimated.position.y_val,3)  )
        return current_state_of_target_car
    
    def get_state_of_adversarial(self, adversarial_car_state):
        current_state_of_adversarial_car = []
        current_state_of_adversarial_car.append(  round(adversarial_car_state.kinematics_estimated.position.x_val,3)  )
        current_state_of_adversarial_car.append(  round(adversarial_car_state.kinematics_estimated.position.y_val,3)  )
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
    def get_reward(self, action):
        delta_steering = 0
        for i in range( 2 , (2+self.num_of_steering) ):
            delta_steering += abs(action[i])
        reward_a = (delta_steering)/self.num_of_steering

        done,success = False, False



        reward_c = 0
        if (self.collision_info_1 == True) and (self.collision_info_2 == True):
                reward_c = 1
                done = True
                success = True

        reward = reward_c - reward_a

        return reward,done,success
#----------------------------------------------------------
#         6. observation & step for learning
#----------------------------------------------------------
    def observation(self):
        while(True):
            try:
                if not(self.collision_info_1): self.collision_info_1 = self.car.simGetCollisionInfo("A_Target").has_collided
                if not(self.collision_info_2): self.collision_info_2 = self.car.simGetCollisionInfo("B_Adversarial").has_collided

        
                target_car_state = self.car.getCarState("A_Target")
                adversarial_car_state = self.car.getCarState("B_Adversarial")

                A_state = self.get_state_of_target(target_car_state)
                B_state = self.get_state_of_adversarial(adversarial_car_state)
                break
            except:
                print("AIRSIM ERROR_07 : request failed")

        # state = A_state + B_state + M
        state = A_state + B_state

        return state
    
    def step(self, action):
        self.reset(action[0], action[1])

        observations = []
        for i in range( 2 , (2+self.num_of_steering) ):
            start_time = time.time()
        
            steering = float(action[i]) 
            throttle = float(action[-1])
            self.set_car_control_of_adversarial(throttle, steering, 0)
            self.set_car_control_of_target(0.6, 0)
            self.set_car_control_of_front(0.6, 0)

            while(1):
                observations.append(self.observation())
                time.sleep(0.1)
                if (time.time()-start_time) > 3 :
                    break

        self.set_car_control_of_adversarial(0, 0, 1)
        self.set_car_control_of_target(0, 1)
        self.set_car_control_of_front(0, 1)
        time.sleep(1)
        self.car.reset()

        reward, done, success = self.get_reward(action)

        return observations, reward, done, success

if __name__ == "__main__":
    k = ENV()

    action  = [0,-10,0.2,-0.4,0.1,0.6]
    observations = k.step(action)
    print(observations)
    