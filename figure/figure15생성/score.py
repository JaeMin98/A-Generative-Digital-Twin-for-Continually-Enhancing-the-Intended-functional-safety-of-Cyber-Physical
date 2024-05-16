import matplotlib.pyplot as plt
import numpy as np

# txt 파일에서 데이터를 읽어 리스트로 반환하는 함수
def read_data_from_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 각 줄을 float 형으로 변환하여 리스트에 추가
            data.append(float(line.strip()))
    return data


# EMA를 계산하는 함수
def calculate_ema(data, alpha=0.01):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
    return ema

def plot_data(data, ylabel, title, figname):
    plt.figure(figsize=(18.25, 2.5))  # 가로로 긴 그래프 크기 설정

    # 원래 데이터 플롯 (연한 색)
    plt.plot(data, label="score", color="r", alpha=0.3)

    # EMA 데이터 플롯 (찐한 색)
    ema_data = calculate_ema(data)
    plt.plot(ema_data, label="score_smooth", color="r")

    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=20)

    plt.xlim([0, len(data)])  # X축의 범위 설정
    plt.xticks(np.arange(0, len(data), 5000), fontsize=15)
    plt.ylim([-300, -100])  # Y축의 범위 설정
    plt.yticks(fontsize=15)

    plt.legend(loc='lower left', fontsize=15)
    plt.grid(True)  # 그리드 추가
    # plt.savefig(figname, bbox_inches='tight', transparent=True)
    plt.savefig(figname, bbox_inches='tight')

# 파일 경로 - 이 부분은 실제 파일 경로로 변경해야 합니다.
file_path = 'episode_reward.txt'
data = read_data_from_txt(file_path)
plot_data(data, "Score", "(a) The score of the adversarial agent during training (lr=0.001)", "figure15_HighLR_score.png")
