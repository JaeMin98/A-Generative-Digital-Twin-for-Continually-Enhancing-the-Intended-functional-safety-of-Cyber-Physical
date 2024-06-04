import random
import matplotlib.pyplot as plt
import seaborn as sns

temp = []

front_throttle_value = 0.65
for i in range(100000):
    change = (random.random() - 0.5) * 0.1
    front_throttle_value += change

    # 극단적인 값에 도달했을 때의 처리를 변경
    if front_throttle_value > 0.90:
        front_throttle_value -= change  # 최대값을 넘으면 변화량을 빼서 조정
    elif front_throttle_value < 0.30:
        front_throttle_value -= change  # 최소값을 넘으면 변화량을 빼서 조정

    temp.append(front_throttle_value)

sns.distplot(temp)
plt.show()
