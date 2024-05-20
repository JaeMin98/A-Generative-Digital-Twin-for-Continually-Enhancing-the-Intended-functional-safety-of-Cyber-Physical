import airsim
import numpy as np

# AirSim 클라이언트 생성
client = airsim.CarClient()
client.confirmConnection()

#x
#-600~600

#y
#-60~60

# 위치 설정 (x, y, z)
position = airsim.Vector3r(600, 0, 1)

# 회전 설정 (w, x, y, z) - 기본 회전
orientation = airsim.Quaternionr(0, 0, 0, 1)

# Pose 객체 생성
pose = airsim.Pose(position, orientation)

# 객체의 pose 설정
client.simSetObjectPose("B_Adversarial", pose, True)