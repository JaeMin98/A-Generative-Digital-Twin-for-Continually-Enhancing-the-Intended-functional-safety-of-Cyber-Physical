import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 천의 크기와 점의 간격 정의
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)

# 구슬의 영향을 모델링하는 함수
def ball_effect(x, y, ball_center, ball_radius):
    distance = np.sqrt((x - ball_center[0])**2 + (y - ball_center[1])**2)*0.5
    return np.exp(-distance**2 / ball_radius**2)

# 구슬의 위치와 크기 설정
ball_center = [0, 0]
ball_radius = 3

# 천의 늘어짐 계산
for i in range(len(x)):
    for j in range(len(y)):
        z[i, j] = -ball_effect(x[i, j], y[i, j], ball_center, ball_radius)

# 3D plot 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, -z, cmap='turbo')

# 라벨과 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ticks=[]
for i in range(20):
    ticks.append(i*0.5) 
ax.set_zticks(ticks)

ax.set_zlim([0, 1.7])

# 플롯 보이기
plt.show()
