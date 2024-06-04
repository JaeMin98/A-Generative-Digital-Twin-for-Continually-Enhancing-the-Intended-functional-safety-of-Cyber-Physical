import matplotlib.pyplot as plt
import numpy as np

# x 값을 정의합니다.
x_range1 = np.linspace(-40, -20, 100, endpoint=False)
x_range2 = np.linspace(-20, 0, 100, endpoint=False)
x_range3 = np.linspace(0, 20, 100, endpoint=False)
x_range4 = np.linspace(20, 40, 100, endpoint=True)

# 각 조건에 맞는 y 값들을 계산합니다.
y_range1 = np.array([-35 for _ in x_range1])
y_range2 = x_range2
y_range3 = -x_range3
y_range4 = np.array([-35 for _ in x_range4])

# y2와 y3 값을 정의합니다.
y2 = np.array([20 for _ in x_range2])
y3 = np.array([-20 for _ in x_range3 if -3 < _ < 3])

# 그래프를 그립니다.
plt.figure(figsize=(10, 4))

# for i in range(8):
#     plt.axhline(y=-40+10*i, color='gray', linestyle='--', alpha=0.2, linewidth=1)
#     plt.axvline(x=-40+10*i, color='gray', linestyle='--', alpha=0.2, linewidth=1)

# y2 그래프를 그립니다. (수평선)
plt.axhline(y=20, color='green', linestyle='-')

plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.axvline(x=-30, color='black', linestyle='-', linewidth=1)

plt.plot([-20,-20],[-20,-35], color='black', linestyle='--', alpha=0.7, linewidth=1)
plt.plot([20,20],[-20,-35], color='black', linestyle='--', alpha=0.7, linewidth=1)

# y1 그래프 부분을 그립니다.
plt.plot(x_range1, y_range1, 'r', lw=1)
plt.plot(x_range2, y_range2, 'b', lw=1)
plt.plot(x_range3, y_range3, 'b', lw=1)
plt.plot(x_range4, y_range4, 'r', lw=1)


# 눈금을 없앱니다.
plt.xticks([])
plt.yticks([])

# 축의 이름을 설정합니다.
# plt.xlabel('D_front')
# plt.ylabel('Score')

# y축의 추가적인 이름을 설정합니다. 수직선을 추가하여 'Cs'와 '-Cb'를 표시합니다.
# plt.text(0, 20, 'Cs', ha='center', va='bottom')
# plt.text(0, -20, '-Cb', ha='center', va='top')

plt.gca().spines['right'].set_visible(False) #오른쪽 테두리 제거
plt.gca().spines['top'].set_visible(False) #위 테두리 제거
plt.gca().spines['left'].set_visible(False) #왼쪽 테두리 제거
plt.gca().spines['bottom'].set_visible(False) #왼쪽 테두리 제거

plt.xlim(-40,40)
plt.ylim(-40,40)

# 그래프를 저장합니다.
plt.savefig('custom_graph.png', format='png', bbox_inches='tight',transparent=True)
