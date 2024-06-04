import setup_path
import airsim
import numpy as np
import random
import time
from gym.spaces import Box
import math

class ENV():
    def __init__(self):
    # define state and action space
        self.action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float)
        self.observation_space_size = 11


        self.action_space_adv = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float)
        self.observation_space_size_adv = 7

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
            
        self.steering_scale = 0.15

    # for (get_state_of_target)
        self.state_size = 11

    # for (get_reward)
        self.distance_threshold = 20
        self.reward_scale_factor = 100
        self.distance_initial_value = 20
        self.dist_reward_scale = 0.1

    def reset(self):
        self.car.reset()
        self.set_car_control_of_target(0.6, 0)
        self.set_car_control_of_adversarial(0.6, 0, 0)
        self.set_car_control_of_front(0.6, 0)
        time.sleep(2)

        state, _, __, ___, _____ = self.observation()

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
        elif throttle < 0.25: throttle = 0.25

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
        front_throttle_value = self.front_car_controls.throttle

        change = (random.random() - 0.5) * 0.1
        front_throttle_value += change

        # 극단적인 값에 도달했을 때의 처리를 변경
        if front_throttle_value > 0.90:
            front_throttle_value -= change  # 최대값을 넘으면 변화량을 빼서 조정
        elif front_throttle_value < 0.30:
            front_throttle_value -= change  # 최소값을 넘으면 변화량을 빼서 조정

        self.front_car_controls.throttle = front_throttle_value
        self.car.setCarControls(self.front_car_controls, "C_Front")


    def get_reward(self, state, target_car_state, adversarial_car_state):
        target_car_x = target_car_state.kinematics_estimated.position.x_val
        target_car_y = target_car_state.kinematics_estimated.position.y_val
        adv_car_x = adversarial_car_state.kinematics_estimated.position.x_val
        adv_car_y = adversarial_car_state.kinematics_estimated.position.y_val


        U_distance = math.sqrt( (adv_car_x - target_car_x) ** 2 + (adv_car_y - target_car_y) ** 2 )

        distance = min(state[0:9])

        if distance >= 20 : reward = -((distance-20)/10)
        else :  reward = ((distance-20)/10)

        done = False
        success = False

        if distance >= 39:
            done = True
            reward -= 100
        elif distance <= 2:
            done = True
            reward -= 100
            
        if target_car_x >= 100:
            done = True
            success = True
            reward += 250


        if(U_distance > 20):
            done = True
            success = True
            reward += 250           


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
            reward -= 0
            print("ENDCODE : COLLISION_01")
            
        distance_x = adv_car_x - target_car_x -3
        if (collision_info_1.has_collided == True) and (collision_info_2.has_collided == True):
            # not ROI collision
            if distance_x < 2.5:
                done = True
                success = True
                print("ENDCODE : COLLISION_02, distance_X : " + str(distance_x))
            
            else: # ROI for collision
                reward -= 200
                done = True
                success = True
                print("ENDCODE : COLLISION_03, distance_X : " + str(distance_x))


        return reward,done,success, target_car_x

    def observation(self):

        target_car_state = self.car.getCarState("A_Target")
        adversarial_car_state = self.car.getCarState("B_Adversarial")

        state = self.get_state_of_target(target_car_state)
        reward, done, success, distance_X = self.get_reward(state,target_car_state,adversarial_car_state)

        return state, reward, done, success, distance_X

    def step(self, action):
        if( 0.1 <= action[1] ): Is_brake = 0
        else: Is_brake = 1
        
        self.set_car_control_of_target(float(action[0]), Is_brake)

#----------------------------------------------------------
#               methods for Adversarial agent
#----------------------------------------------------------
    def get_state_of_front_for_Adv(self, front_car_state):
        current_state_of_front_car = []
        current_state_of_front_car.append(  round(front_car_state.kinematics_estimated.position.x_val,3)  )
        current_state_of_front_car.append(  round(front_car_state.speed, 3)  )

        return current_state_of_front_car

    def get_state_of_target_for_Adv(self, target_car_state):
        current_state_of_target_car = []
        current_state_of_target_car.append(  round(target_car_state.kinematics_estimated.position.x_val,3)  )
        current_state_of_target_car.append(  round(target_car_state.speed, 3)  )

        return current_state_of_target_car


    def get_state_of_adversarial_for_Adv(self, adversarial_car_state):
        current_state_of_adversarial_car = []
        current_state_of_adversarial_car.append(  round(adversarial_car_state.kinematics_estimated.position.x_val,3)  )
        current_state_of_adversarial_car.append(  round(adversarial_car_state.kinematics_estimated.position.y_val,3)  )
        current_state_of_adversarial_car.append(  round(adversarial_car_state.speed, 3)  )

        return current_state_of_adversarial_car

    def set_car_control_of_adversarial(self, throttle, steering, brake):
        self.adversarial_car_controls.throttle = throttle
        self.adversarial_car_controls.steering = steering
        self.adversarial_car_controls.brake = brake

        try:
            self.car.setCarControls(self.adversarial_car_controls, "B_Adversarial")
        except:
            print("AIRSIM ERROR_03 : request failed")

    def get_Adv_state(self):

        while(True):
            try:
                target_car_state = self.car.getCarState("A_Target")
                adversarial_car_state = self.car.getCarState("B_Adversarial")
                front_car_state = self.car.getCarState("C_Front")

                A_state = self.get_state_of_target_for_Adv(target_car_state)
                B_state = self.get_state_of_adversarial_for_Adv(adversarial_car_state)
                C_state = self.get_state_of_front_for_Adv(front_car_state)
                break
            except:
                print("AIRSIM ERROR_07 : request failed")          

        state = A_state + B_state + C_state
        return state

    def step_for_Adv(self, action):
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

        