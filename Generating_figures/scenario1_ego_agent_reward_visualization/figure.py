import matplotlib.pyplot as plt
import numpy as np

# y1 함수를 정의합니다.
def y1(x):
    if x < -20:
        return -20
    elif -20 <= x < 0:
        return x
    elif 0 <= x < 20:
        return -x
    else:
        return -20

# x 범위를 설정합니다.
x = np.linspace(-30, 30, 400)

# y1, y2, y3 값을 계산합니다.
y1_values = [y1(xi) for xi in x]
y2_values = [20 for _ in x]
y3_values = [-20 if -3 < xi < 3 else None for xi in x]

# 그래프를 그립니다.
plt.figure(figsize=(8, 6))
plt.plot(x, y1_values, label='y1')
plt.plot(x, y2_values, label='y2', linestyle='--')
plt.plot(x, y3_values, label='y3', linestyle=':')

# 축 이름을 설정합니다.
plt.xlabel('D_Front')
plt.ylabel('Reward')

# 눈금을 없앱니다.
plt.xticks([])
plt.yticks([])

# 범례를 표시합니다.
plt.legend()

# 그래프를 저장합니다.
plt.savefig('graph.png', format='png')

# 그래프를 보여줍니다.
plt.show()
