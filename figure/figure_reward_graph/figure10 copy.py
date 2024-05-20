import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 3D plot 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 라벨과 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('')


ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([0, 0.1])

ticks = []
for i in range(20):
    ticks.append(i*0.5) 
ax.set_zticks([])

ax.set_xticks([-10, -5, 0 , 5 , 10])
ax.set_yticks([-10, -5, 0 , 5 , 10])

# 플롯 보이기
plt.show()
