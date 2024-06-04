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

def calculate_distance(x, y, a, b):
    distance = math.sqrt((x - a) ** 2 + (y - b) ** 2)
    return distance

def get_state_of_target(target_car_state):
    current_state_of_target_car = []
    current_state_of_target_car.append(  round(target_car_state.kinematics_estimated.position.x_val,3)  )
    current_state_of_target_car.append(  round(target_car_state.kinematics_estimated.position.y_val,3)  )
    return current_state_of_target_car

def get_state_of_adversarial(adversarial_car_state):
    current_state_of_adversarial_car = []
    current_state_of_adversarial_car.append(  round(adversarial_car_state.kinematics_estimated.position.x_val,3)  )
    current_state_of_adversarial_car.append(  round(adversarial_car_state.kinematics_estimated.position.y_val,3)  )
    return current_state_of_adversarial_car

car = airsim.CarClient()
car.confirmConnection()
car.enableApiControl(True, "A_Target")
car.enableApiControl(True, "B_Adversarial")
car.enableApiControl(True, "C_Front")

car.reset()
target_car_state = car.getCarState("A_Target")
adversarial_car_state = car.getCarState("B_Adversarial")
front_car_state = car.getCarState("C_Front")

initial_state_A = [0,-3]
initial_state_B = [-3,-10]
print(initial_state_A, initial_state_B)

time.sleep(2)

position = airsim.Vector3r(0, 0, -3)
orientation = airsim.Quaternionr(0, 0, 0, 1)
pose = airsim.Pose(position, orientation)
car.simSetObjectPose("A_Target", pose, True)

position = airsim.Vector3r(20, 20, -3)
orientation = airsim.Quaternionr(0, 0, 0, 1)
pose = airsim.Pose(position, orientation)
car.simSetObjectPose("B_Adversarial", pose, True)

time.sleep(2)

target_car_state = car.getCarState("A_Target")
adversarial_car_state = car.getCarState("B_Adversarial")
front_car_state = car.getCarState("C_Front")

A_state = get_state_of_target(target_car_state)
B_state = get_state_of_adversarial(adversarial_car_state)
for i in range(2):
    A_state[i] += initial_state_A[i]
    B_state[i] += initial_state_B[i]

print(A_state, B_state)
print(calculate_distance(A_state[0], A_state[1], B_state[0], B_state[1]))