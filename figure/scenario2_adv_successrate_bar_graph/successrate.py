

import matplotlib.pyplot as plt
import numpy as np

# txt 파일에서 데이터를 읽어 리스트로 반환하는 함수
def read_data_from_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 'True'와 'False' 값을 boolean 형으로 변환하여 리스트에 추가
            data.append(line.strip() == 'True')
    return data

# 데이터를 400개 단위로 나누어 각 부분의 성공률을 계산하는 함수
def calculate_success_rate(data, chunk_size=400):
    success_rates = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        if chunk:  # 빈 청크를 피하기 위한 검사
            success_rate = sum(chunk) / len(chunk) * 100  # 성공률 계산
            success_rates.append(success_rate)
    return success_rates

# 성공률을 막대 그래프로 나타내는 함수
def plot_success_rate(success_rates):
    plt.figure(figsize=(9.6, 4.0))  # 그래프 크기 설정
    labels = [f"{i*400} to\n{i*400 + 400}" for i in range(len(success_rates))]

    plt.bar(range(0,6), [100,100,100,100,100,100], color="#ff7f0e", label="Infraction inducing rate", tick_label=labels)
    plt.bar(range(len(success_rates)), success_rates, color="#1f77b4", label="Infraction avoiding rate", tick_label=labels)

    x_positions = range(len(success_rates))  # 막대 그래프의 x 위치
    # 각 막대의 끝 점에 빨간색 점 표시
    for i, rate in enumerate(success_rates):
        plt.plot(i, rate, 'ro', markersize=7)  # 'ro'는 빨간색 원형 마커를 의미

    # 빨간색 선으로 점들을 연결
    plt.plot(x_positions, success_rates, linestyle='--', color="r", linewidth=2.5)  # 'r-'는 빨간색 실선을 의미

    plt.xlabel('Episode', fontsize=18)
    plt.ylabel('percentage (%)', fontsize=18)
    plt.title('', fontsize=20)
    plt.xticks(np.arange(len(success_rates)), fontsize=18)
    plt.yticks(fontsize=18)
    # plt.grid(axis='y', linestyle='--')
    plt.legend(loc='upper left', fontsize=15)

    plt.ylim([0, 105])  # Y축의 범위 설정
    plt.yticks(np.arange(0, 101, 20), fontsize=18)

    plt.savefig("figure14_success_rate.png", bbox_inches='tight')

# 파일 경로 - 이 부분은 실제 파일 경로로 변경해야 합니다.
file_path = 'success.txt'
data = read_data_from_txt(file_path)[:2100]
success_rates = calculate_success_rate(data)
plot_success_rate(success_rates)