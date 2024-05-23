from operator import truediv
import setup_path

import csv
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
        self.action_space = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float_)
        self.observation_space_size = 7

        self.action_space_of_ego = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float_)
        self.observation_space_size_of_ego = 11

    # base parameter setting
        self._max_episode_steps = 256

        while(1):
            try:
                self.car = airsim.CarClient()
                self.car.confirmConnection()

                self.car.enableApiControl(True, "A_Target")
                self.car.enableApiControl(True, "B_Adversarial")
                self.car.enableApiControl(True, "C_Front")
                

                self.target_car_controls = airsim.CarControls("A_Target")
                self.adversarial_car_controls = airsim.CarControls("B_Adversarial")
                self.front_car_controls = airsim.CarControls("C_Front")
                break
            except:
                print("AIRSIM ERROR_01 : request failed")

        self.steering_scale = 0.15
        self.set_initial_position()

    # for (get_state_of_target)
        self.state_size = 7

    # for (get_reward)
        self.distance_threshold = 20
        self.reward_scale_factor = 100
        self.distance_initial_value = 20
        self.dist_reward_scale = 0.1

        self.figure_data = []

    def set_initial_position(self):
        json_file_path = "/home/smartcps/Documents/AirSim/settings.json"
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        self.initial_state_A = [data['Vehicles']['A_Target']['X'], data['Vehicles']['A_Target']['Y']]
        self.initial_state_B = [data['Vehicles']['B_Adversarial']['X'], data['Vehicles']['B_Adversarial']['Y']]
        self.initial_state_C = [data['Vehicles']['C_Front']['X'], data['Vehicles']['C_Front']['Y']]

        


#----------------------------------------------------------
#                     2. reset 
#----------------------------------------------------------
    def reset(self):
        self.car.reset()

        # position = airsim.Vector3r(-3, -10, -3)
        # orientation = airsim.Quaternionr(0, 0, 0, 1)
        # pose = airsim.Pose(position, orientation)
        # self.car.simSetObjectPose("B_Adversarial", pose, True)
        # time.sleep(2)

        self.figure_data = []
        self.set_car_control_of_target(0.6, 0)
        self.set_car_control_of_adversarial(0.6, 0, 0)
        self.set_car_control_of_front(0.6, 0)
        time.sleep(2)

        state, _, __, ___, _____ = self.observation()

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

        try:
            self.car.setCarControls(self.front_car_controls, "C_Front")
        except:
            print("AIRSIM ERROR_05 : request failed")

#----------------------------------------------------------
#                 5. reward function 
#----------------------------------------------------------

    def get_reward(self, adversarial_car_state, target_car_state):
        target_car_x = target_car_state.kinematics_estimated.position.x_val
        target_car_y = target_car_state.kinematics_estimated.position.y_val
        adv_car_x = adversarial_car_state.kinematics_estimated.position.x_val
        adv_car_y = adversarial_car_state.kinematics_estimated.position.y_val


        done = False
        success = False

        distance = math.sqrt( (adv_car_x - target_car_x) ** 2 + (adv_car_y - target_car_y) ** 2 )
        distance_x = adv_car_x - target_car_x + 4

        if( distance > 10 ) :
            print("ENDCODE : OUT OF BOUNDARY, distance : " + str(distance))
            done = True
            out_of_bound = -100
        else : 
            out_of_bound = 0

        reward = -(distance/5) + out_of_bound

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
            if distance_x < 2.5:
                reward += 200
                done = True
                success = True
                print("ENDCODE : COLLISION_02, distance_X : " + str(distance_x))
            
            else: # ROI for collision
                reward += 200
                done = True
                success = True
                print("ENDCODE : COLLISION_03, distance_X : " + str(distance_x))

        return reward,done,success


#----------------------------------------------------------
#         6. observation & step for learning
#----------------------------------------------------------
    def write_figure_data(self, file_path):
        column_names = ['IsCrash', 'A.X', 'A.Y', 'A.Pitch', 'A.Roll', 'A.Yaw', 'B.X', 'B.Y', 'B.Pitch', 'B.Roll', 'B.Yaw', 'C.X', 'C.Y', 'C.Pitch', 'C.Roll', 'C.Yaw']

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)

            for item in self.figure_data:
                writer.writerow(item)

    def update_figure_data(self, vehicle_states):
        initial_values = [self.initial_state_A, self.initial_state_B, self.initial_state_C]
        temp_data = []

        if(self.collision_info_1 and self.collision_info_2): IsCrash = True
        else: IsCrash = False
        temp_data.append(IsCrash)

        for i in range(len(vehicle_states)):
            position = vehicle_states[i].kinematics_estimated.position
            x, y = position.x_val + initial_values[i][0], position.y_val + initial_values[i][1]

            orientation = vehicle_states[i].kinematics_estimated.orientation
            pitch, roll, yaw = airsim.to_eularian_angles(orientation)

            temp_data += [x, y, pitch, roll, yaw]

        self.figure_data.append(temp_data)

    def observation(self):
        while(True):
            try:
                target_car_state = self.car.getCarState("A_Target")
                adversarial_car_state = self.car.getCarState("B_Adversarial")
                front_car_state = self.car.getCarState("C_Front")

                A_state = self.get_state_of_target(target_car_state)
                B_state = self.get_state_of_adversarial(adversarial_car_state)
                C_state = self.get_state_of_front(front_car_state)

                for i in range(len(self.initial_state_A)):
                    A_state[i] += self.initial_state_A[i]
                    B_state[i] += self.initial_state_B[i]
                    C_state[i] += self.initial_state_C[i]
                break
            except:
                print("AIRSIM ERROR_07 : request failed")          

        state = A_state + B_state + C_state

        reward, done, success = self.get_reward(adversarial_car_state, target_car_state)

        E_reward, E_done, E_success = self.get_Ego_reward(target_car_state, front_car_state)

        self.update_figure_data(success, (target_car_state, adversarial_car_state, front_car_state))

        return state, reward, done, success, E_reward

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
        
        while(True):
            try:
                self.set_car_control_of_target(float(action[0]), Is_brake)
                break
            except:
                print("AIRSIM ERROR_09 : request failed")
        
    def get_Ego_reward(self, target_car_state, front_car_state):
        target_car_x = target_car_state.kinematics_estimated.position.x_val
        front_car_x = front_car_state.kinematics_estimated.position.x_val
        
        distance = front_car_x - target_car_x + self.distance_initial_value
        distance_gap = float(abs(distance - self.distance_threshold))

        reward = -(distance_gap/10)
        done = False
        success = False

        if distance_gap >= self.distance_threshold -4.5:
            done = True
            reward -= 100

        if target_car_x >= 300:
            done = True
            success = True
            reward += 250

        return reward,done,success

if __name__ == "__main__":
    k = ENV()
    k.reset()


    k.set_car_control_of_adversarial(0.7, -0.15, 0)