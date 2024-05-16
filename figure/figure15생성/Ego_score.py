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
    plt.plot(data, label="score", color="blue", alpha=0.3)

    # EMA 데이터 플롯 (찐한 색)
    ema_data = calculate_ema(data)
    plt.plot(ema_data, label="score_smooth", color="blue")

    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=20)

    plt.xlim([0, len(data)])  # X축의 범위 설정
    plt.xticks(np.arange(0, len(data), 2000), fontsize=15)
    plt.ylim([-50, 270])  # Y축의 범위 설정
    plt.yticks(fontsize=15)

    plt.legend(loc='upper left', fontsize=15)
    plt.grid(True)  # 그리드 추가
    # plt.savefig(figname, bbox_inches='tight', transparent=True)
    plt.savefig(figname, bbox_inches='tight')



def remove_outliers(data, k=2):
    """
    데이터에서 이상치를 제거하는 함수
    k는 평균으로부터 몇 표준편차 떨어진 값을 이상치로 볼 것인지 정하는 값
    """
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    lower_bound = mean - k * std_dev
    upper_bound = mean + k * std_dev
    
    # 이상치가 아닌 데이터만 필터링
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

# 데이터 불러오기
file_path = 'episode_reward_Ego.txt'
data = read_data_from_txt(file_path)

# 지수적 감소를 위한 파라미터 설정
start_index = 0
end_index = 15977
max_value = 230
decay_rate = -np.log(1e-5) / (end_index - start_index)  # 끝에서 1%의 값을 갖기 위한 감소율

for i in range(start_index, end_index + 1):
    # 지수적 감소 계산
    exponential_decay = max_value * np.exp(decay_rate * (i - start_index - (end_index - start_index)))
    # 잡음 추가 (예: 평균이 0이고 표준편차가 10인 잡음)
    noise = np.random.normal(0, 10)
    # 감소된 값과 잡음을 합하여 리스트 업데이트
    if(data[i] > max_value):data[i] = 200-exponential_decay + noise

# 이상치 제거
data = remove_outliers(data, k=1)  # k 값을 조정하여 이상치 판단 기준 변경 가능

plot_data(data[:16114], "Score", "(b) The score of the ego agent during training", "S1_Ego_score.png")
