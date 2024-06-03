from operator import truediv
import setup_path
import airsim
import numpy as np
import random
import time
import math
import json
from gym.spaces import Box
import csv

import config
from sac import SAC


class ENV():
#----------------------------------------------------------
#              1. initialization parameters
#----------------------------------------------------------
    def __init__(self):
    # define state and action space (전진, 회전, 브레이크)
        # steering_low, steering_high = -0.15, 0.15
        # low = np.array([-20.0, -20.0, steering_low, steering_low, steering_low, 0.5])
        # high = np.array([-10.0, -10.0, steering_high, steering_high, steering_high, 1.0])
        # self.action_space = Box(low=low, high=high,shape=(6,), dtype=np.float_)

        yaw_low, yaw_high = 0.0, 0.5
        low = np.array([-11.0, -16.0, yaw_low, 0.5])  #A.Y = -3
        high = np.array([-8.0, -13.0, yaw_high, 1.0])
        self.action_space = Box(low=low, high=high,shape=(4,), dtype=np.float_)

    # self.action_space = Box(low=0.0, high=1.0, shape=(6,), dtype=np.float_)
        self.observation_space_size = 5
        self.num_of_steering = 3

        self.action_space_of_ego = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float_)
        self.observation_space_size_of_ego = 11

        self.figure_data = []
        self.IsCollision = False
        self.set_initial_position()
        self.set_ego_agent()

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

    
    def set_initial_position(self):
        json_file_path = "/home/smartcps/Documents/AirSim/settings.json"
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        self.initial_state_A = [data['Vehicles']['A_Target']['X']-1.45, data['Vehicles']['A_Target']['Y']]
        self.initial_state_B = [data['Vehicles']['B_Adversarial']['X']-1.45, data['Vehicles']['B_Adversarial']['Y']]
        self.initial_state_C = [data['Vehicles']['C_Front']['X']-1.45, data['Vehicles']['C_Front']['Y']]

    def set_ego_agent(self):
        parser = config.parser
        args = parser.parse_args()
        Ego_args = args
        Ego_args.hidden_size = 128

        self.Ego_agent = SAC(self.observation_space_size_of_ego, self.action_space_of_ego, Ego_args)
        self.Ego_agent.load_checkpoint("ego_agent_model/ego_agent.tar")

#----------------------------------------------------------
#                     2. reset
#----------------------------------------------------------
    def reset(self, x,y,yaw):
        self.car.reset()
        self.set_position(float(x),float(y),float(yaw))

        self.set_car_control_of_target(0.6, 0)
        self.set_car_control_of_adversarial(0.6, 0, 0)
        self.set_car_control_of_front(0.6, 0)
        time.sleep(2)
    
    def set_position(self, x, y , yaw):
        position = airsim.Vector3r(x, y, -3)
        orientation = airsim.Quaternionr(0, 0, yaw, 1)
        pose = airsim.Pose(position, orientation)
        self.car.simSetObjectPose("B_Adversarial", pose, True)

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
    def get_reward(self, action, observations):
        done,success = False, False

        reward_c = 0
        if (self.IsCollision == True):
                print("################### Collision ###################")
                reward_c = 1
                done = True
                success = True

        distance_list = []
        for item in observations:
            distance_list.append(item[-1])

        reward_m = -(sum(distance_list)/len(distance_list))

        reward = reward_m + 10*reward_c

        return reward,done,success
#----------------------------------------------------------
#         6. observation & step for learning
#----------------------------------------------------------

    def calculate_distance(self, x, y, a, b):
        distance = math.sqrt((x - a) ** 2 + (y - b) ** 2)
        return distance

    def write_figure_data(self, file_path):
        column_names = ['IsCollision', 'A.X', 'A.Y', 'A.Pitch', 'A.Roll', 'A.Yaw', 'B.X', 'B.Y', 'B.Pitch', 'B.Roll', 'B.Yaw', 'C.X', 'C.Y', 'C.Pitch', 'C.Roll', 'C.Yaw']

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)

            for item in self.figure_data:
                writer.writerow(item)

    def update_figure_data(self, vehicle_states):
        initial_values = [self.initial_state_A, self.initial_state_B, self.initial_state_C]
        temp_data = []

        temp_data.append(self.IsCollision)

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
                #State Check
                target_car_state = self.car.getCarState("A_Target")
                adversarial_car_state = self.car.getCarState("B_Adversarial")
                front_car_state = self.car.getCarState("C_Front")

                A_state = self.get_state_of_target(target_car_state)
                B_state = self.get_state_of_adversarial(adversarial_car_state)


                for i in range(len(self.initial_state_A)):
                    A_state[i] += self.initial_state_A[i]
                    B_state[i] += self.initial_state_B[i]

                #Collision Check
                distance_x = (B_state[0] - A_state[0])
                
                try:
                    collision_info_1 = self.car.simGetCollisionInfo("A_Target")
                    collision_info_2 = self.car.simGetCollisionInfo("B_Adversarial")
                    # collision_info_3 = self.car.simGetCollisionInfo("C_Front")
                except:
                    collision_info_1 = False
                    collision_info_2 = False
                    # collision_info_3 = False
                    print("AIRSIM ERROR_06 : request failed")

                # if (collision_info_3.has_collided == True) and (collision_info_2.has_collided == True): print("ENDCODE : COLLISION_01")
                if (collision_info_1.has_collided == True) and (collision_info_2.has_collided == True) and (self.IsCollision == False):
                    self.IsCollision = True
                    if distance_x >= 1.05: # ROI collision
                        self.Is_ROI_Collision = True
        
                self.update_figure_data((target_car_state, adversarial_car_state, front_car_state))
                break
            except:
                print("AIRSIM ERROR_07 : request failed")


        # state = A_state + B_state + M
        state = A_state + B_state

        distance = self.calculate_distance(A_state[0], A_state[1], B_state[0], B_state[1])
        state.append(distance)

        return state
    
    def step(self, action):
        self.figure_data = []
        self.IsCollision = False
        self.Is_ROI_Collision = False
        self.reset(action[0], action[1], action[2])

        observations = []
        
        start_time = time.time()
        throttle = float(action[-1])
        self.set_car_control_of_adversarial(throttle, 0, 0)
        self.set_car_control_of_target(0.6, 0)
        self.set_car_control_of_front(0.6, 0)

        count = 0
        while(1):
            observations.append(self.observation())
            time.sleep(0.1)
            if (time.time()-start_time) > 9 :
                break

            if( (count % 30) == 0) and count != 0:
                self.set_car_control_of_front_random()
            Ego_state = self.get_Ego_state()
            self.step_for_Ego(self.Ego_agent.select_action(Ego_state))

            count += 1

        self.set_car_control_of_adversarial(0, 0, 1)
        self.set_car_control_of_target(0, 1)
        self.set_car_control_of_front(0, 1)
        time.sleep(1)
        self.car.reset()

        reward, done, success = self.get_reward(action, observations)

        return observations, reward, done, success, self.Is_ROI_Collision #here
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

if __name__ == "__main__":
    k = ENV()

    action  = [0, -10, 0.2, -0.4, 0.1, 0.6]
    observations = k.step(action)
    print(observations)
    