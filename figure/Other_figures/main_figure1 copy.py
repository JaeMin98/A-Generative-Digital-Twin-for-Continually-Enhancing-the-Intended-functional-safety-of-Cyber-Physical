import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import os
import glob

def chunk_list(input_list, chunk_sizes):
    chunks = []
    start = 0
    for size in chunk_sizes:
        chunks.append(input_list[start:start+size])
        start += size

    # If there are remaining items in the input_list, add them to the last chunk
    if start < len(input_list):
        chunks[-1].extend(input_list[start:])

    return chunks

def compute_ema(data, s, is_chunk, chunk_sizes=[82, 143, 93, 47]):

    if is_chunk :
        chunked_data = chunk_list(data, chunk_sizes=chunk_sizes)
        result = []
        for data in chunked_data:
            ema = [data[0]]
            for t in range(1, len(data)):
                ema_value = (1 - s) * data[t] + s * ema[t-1]
                ema.append(ema_value)
            result += ema
        return result
    else :
        ema = [data[0]]
        for t in range(1, len(data)):
            ema_value = (1 - s) * data[t] + s * ema[t-1]
            ema.append(ema_value)
        return ema

class plot_data():
    def load_csv(self, csv_file_path):
        # 빈 리스트 생성
        column_2_data = []
        column_3_data = []

        # CSV 파일 열기
        with open(csv_file_path, 'r', newline='') as csvfile:
            # CSV 파일 읽기
            csvreader = csv.reader(csvfile)
            
            # 각 행을 반복하면서 2번째와 3번째 열의 데이터 추출
            for row in csvreader:
                if len(row) >= 3:  # 적어도 3개 열이 있는 경우
                    column_2_data.append(row[1])  # 2번째 열 데이터를 리스트에 추가
                    if(row[2]=='Value'):
                        column_3_data.append(row[2])
                    else:
                        column_3_data.append(float(row[2]))  # 3번째 열 데이터를 리스트에 추가
        del column_2_data[0]
        del column_3_data[0]

        column_2_data = list(map(float,column_2_data))
        column_3_data = list(map(float,column_3_data))

        return column_2_data, column_3_data
    

    def plot(self, is_level_plot, csv_list, scale, is_chunck, chunk_size, save_path):
        for i, csv_path in enumerate(csv_list):
            print(csv_path)

            xlabel = "episode"
            ylabel = csv_path.split('tag-')[-1].split('.')[0]
            X_data, Y_data = self.load_csv(csv_path)

            x = X_data
            y = Y_data
            s = scale

            ema_values = compute_ema(y, s, is_chunck, chunk_size)

            Replay_Ratio = csv_path.split('_SAC_')[1].split('_')[0]
            if Replay_Ratio == 'NonCL':
                csv_name = 'SAC w/o automated curriculum scheme'
            else:
                csv_name = 'SAC w/ automated curriculum scheme (ACS), Buffer flushing : True,  Replay ratio : ' + str(str(   round(   (1.0 - float(Replay_Ratio)) , 2)   ))

            fig, ax = plt.subplots(figsize=(42, 8))  # Create a new figure for each plot

            # Plot
            ax.plot(x, y, '-', color='b', linewidth=1.7, alpha=0.35, label='Original Data')
            ax.plot(x, ema_values, '-', color='red', linewidth=1.5, label='Smoothed Data')

            ax.set_xlabel(xlabel, fontsize=25)
            ax.set_ylabel("Score", fontsize=25)
            ax.set_title(csv_name, fontsize=30)
            ax.set_ylim(-300, 100)
            ax.legend(fontsize=20, loc='lower right')
            ax.grid(True)
            ax.tick_params(axis='x', labelsize=23)
            ax.tick_params(axis='y', labelsize=23)

            if(Replay_Ratio == "NonCL"):
                individual_save_path = f"{save_path}_{Replay_Ratio}.png"
            else:
                individual_save_path = f"{save_path}_{str(   round(   (1.0 - float(Replay_Ratio)) , 2)   )}.png"
            plt.savefig(individual_save_path)  # Save the current figure
            plt.close(fig)  # Close the figure to free memory


    def make_folder(self,folder_path):
        # 폴더가 이미 존재하는지 확인
        if not os.path.exists(folder_path):
            # 폴더가 존재하지 않으면 생성
            os.makedirs(folder_path)
        else:
            pass

    def get_file_list(self, folder_path):
        # 현재 작업 디렉토리 가져오기
        current_directory = folder_path
        result = []

        # CSV 파일 검색 및 파일명 리스트 추출
        csv_files = glob.glob(os.path.join(current_directory, "*.csv"))

        # CSV 파일명 리스트 출력
        for csv_file in csv_files:
            csv_file = csv_file.split('\\')[1]
            csv_file = csv_file.split('-tag')[0]
            result.append(csv_file)
            
        return result


    def run(self):
        # CSV 파일을 저장할 리스트를 생성합니다.
        csv_list = []

        # CSV 파일이 있는 디렉토리 경로를 지정합니다.
        directory_path = './score'  # 디렉토리 경로를 변경하세요.

        # 디렉토리 내의 모든 파일 목록을 얻습니다.
        file_list = os.listdir(directory_path)

        # 디렉토리 내의 모든 파일 중에서 CSV 파일만 필터링하여 리스트에 추가합니다.
        for file_name in file_list:
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory_path, file_name)
                csv_list.append(os.path.normpath(file_path))

        # csv_list에는 디렉토리 내의 모든 CSV 파일의 경로가 포함됩니다.
        print(csv_list)

        save_path = "./plot_image/figure1_Score.png"

        # xlabel = "episode"
        # ylabel = csv_path.split('tag-')[-1].split('.')[0]
        # X_data,Y_data = self.load_csv(csv_path[0])
        scale = 0.94

        self.plot(False, csv_list, scale, False, [], save_path)

        
        

if __name__ == "__main__":
   plot = plot_data()
   plot.run()