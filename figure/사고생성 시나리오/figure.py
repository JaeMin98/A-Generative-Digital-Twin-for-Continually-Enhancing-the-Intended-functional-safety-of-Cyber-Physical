import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import csv
import os
import math
from tqdm import tqdm

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_csv(file_path):
    csv_data = []

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            csv_data.append(row)

    del(csv_data[0])

    def convert_row(row):
        if row[0]=='True':IsCrashed = True
        else : IsCrashed = False
        return [IsCrashed] + [float(value) for value in row[1:]]
    csv_data = [convert_row(row) for row in csv_data]

    return (csv_data)

def calculate_yaw(path):
    yaw_angles = []
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        yaw = math.atan2(dy, dx)
        yaw_angles.append(yaw)
    yaw_angles.append(yaw_angles[-1])  # 마지막 yaw 값은 이전 값으로 동일하게 설정
    return np.array(yaw_angles)

def plot_figure(csv_data,figure_name):
    w=15
    h=5
    fig, ax = plt.subplots(figsize=(w, h))
    # ax 테두리 설정
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(20)
#-----------------------------------------------------------------------------------------------------------------------------
    # 자동차 경로 그리기 (빨간색, 파란색, 녹색)   . csv파일 읽어오기. 정규화 진행하기(0~1)
    
    for row in csv_data:
        for i in [1,6,11]:
            if i < len(row):  # 행의 길이를 벗어나지 않는지 확인
                row[i] = row[i]*2.5

        for i in [2,7,12]:
            if i < len(row):  # 행의 길이를 벗어나지 않는지 확인
                row[i] += 3
                row[i] = row[i]*2.0
        
        row[7] -= 4

    red_path = np.array([[row[1], row[2]] for row in csv_data])
    blue_path = np.array([[row[6], row[7]] for row in csv_data])
    green_path = np.array([[row[11], row[12]] for row in csv_data])

    path_line_width = 2.5
    path_line_zorder = 3
    ax.plot(red_path[:, 0], red_path[:, 1], 'r-', linewidth=path_line_width, zorder=path_line_zorder)
    ax.plot(blue_path[:, 0], blue_path[:, 1], 'b-', linewidth=path_line_width, zorder=path_line_zorder)
    ax.plot(green_path[:, 0], green_path[:, 1], 'g-', linewidth=path_line_width, zorder=path_line_zorder)
#-----------------------------------------------------------------------------------------------------------------------------
    red_yaws = calculate_yaw(red_path)
    blue_yaws = calculate_yaw(blue_path)
    green_yaws = calculate_yaw(green_path)

    # 자동차 그리기 함수 (orientation 포함)
    def draw_car(ax, position, yaw, color):
        scale = 2.5
        car_width, car_height = 4 * scale, 3 * scale
        car_zorder = 4
        
        # 변환 설정
        t = transforms.Affine2D().rotate_around(position[0], position[1], yaw) + ax.transData
        car = patches.Rectangle((position[0] - car_width / 2, position[1] - car_height / 2), 
                                car_width, car_height, linewidth=2.5, alpha=0.5, color=color, zorder=car_zorder,
                                transform=t)
        
        ax.add_patch(car)
        ax.plot(position[0], position[1], 'o', color=color, markersize=8, zorder=car_zorder)


    #collision index
    Collision_indexs = []
    for i, row in enumerate(csv_data):
        if row[0] == True:
            Collision_indexs.append(i)
    C_index = min(Collision_indexs)
    

    for i in range(0, len(green_path)):
        if(i == 0) or (i == C_index) or (i % 10 == 0): draw_car(ax, green_path[i], green_yaws[i], 'green')
    for i in range(0, len(red_path)):
        if(i == 0) or (i == C_index) or (i % 10 == 0): draw_car(ax, red_path[i], red_yaws[i], 'red')
    for i in range(0, len(blue_path)):
        if(i == 0) or (i == C_index) or (i % 10 == 0): draw_car(ax, blue_path[i], blue_yaws[i], 'blue')
#-----------------------------------------------------------------------------------------------------------------------------
    # 충돌 지점 그리기
    collision_point = [(csv_data[C_index][1] + csv_data[C_index][6])/2,
                       (csv_data[C_index][2] + csv_data[C_index][7])/2]

    collision_maker_zorder = 5
    ax.plot(collision_point[0], collision_point[1], '*', color='black', markersize=30, zorder=collision_maker_zorder)
    ax.plot(collision_point[0], collision_point[1], '*', color='orange', markersize=20, zorder=collision_maker_zorder)
#-----------------------------------------------------------------------------------------------------------------------------
   # 도로 배경 그리기
    road = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='lightgrey')
    ax.add_patch(road)
    white_line_width = 6
    white_line_zorder = 1
    line = ax.axhline(y=-12.5, color='white', linestyle='dashed', linewidth=white_line_width, zorder=white_line_zorder)
    line.set_dashes([5, 4.1])  # [선의 길이, 간격]
    line = ax.axhline(y=12.5, color='white', linestyle='dashed', linewidth=white_line_width, zorder=white_line_zorder)
    line.set_dashes([5, 4.1])  # [선의 길이, 간격]

#-----------------------------------------------------------------------------------------------------------------------------
    ax.set_xlim(-20, 160)
    ax.set_ylim(-30, 30)
    ax.set_aspect('auto')
    plt.axis('off')
    
    plt.savefig(figure_name, bbox_inches='tight', pad_inches=0.2)




directory = "case2/left_side"

csv_files = []
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        csv_files.append(filename)

# csv_files=['221.csv']
for file in tqdm(csv_files):
    csv_path = os.path.join(directory, file)
    csv_data =  read_csv(csv_path)

    image_directory = os.path.join('image2', directory)
    create_folder_if_not_exists(image_directory)

    image_path = os.path.join('image2', csv_path)
    image_path = image_path[:-4] + '.png'

    plot_figure(csv_data, image_path)


# directory = "case1"

# csv_files = ['759.csv']

# for file in csv_files:
#     csv_path = os.path.join(directory, file)

#     csv_data =  read_csv(csv_path)
#     image_path = os.path.join('image', csv_path)
#     image_path = image_path[:-4] + '.png'
#     plot_figure(csv_data, image_path)