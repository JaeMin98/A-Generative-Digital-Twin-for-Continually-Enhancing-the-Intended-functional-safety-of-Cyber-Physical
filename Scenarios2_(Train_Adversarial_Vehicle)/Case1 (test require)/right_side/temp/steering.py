import airsim
import time

# AirSim 차량 클라이언트 연결
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

# 속도를 0.5로 고정
fixed_speed = 0.5

# steering data 리스트 예제
steering_data = [0.8, 0.1, 0.2, 0.5, -0.6, -0.1, -0.8, -0.2]

try:
    for steering_value in steering_data:
        # 속도 및 조향 값 설정
        car_controls.throttle = fixed_speed
        car_controls.steering = steering_value

        # 차량에 제어 명령 전송
        client.setCarControls(car_controls)

        # 5초 대기
        time.sleep(3)
finally:
    # API 제어 해제
    client.enableApiControl(False)
