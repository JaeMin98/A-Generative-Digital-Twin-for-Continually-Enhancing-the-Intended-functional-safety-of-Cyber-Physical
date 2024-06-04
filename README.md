
# AirSim 설치 가이드 (Linux) 🛠️

## 시스템 요구사항 🖥️

- **OS**: Ubuntu 18.04 / 20.04
- **Hardware**: 8GB RAM, 100GB+ 디스크 공간, NVIDIA GPU (CUDA 10.0+)

## 설치 과정 🛠️

### 1. 의존성 패키지 설치 📦

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libjpeg-dev libpng-dev libtiff-dev libgl1-mesa-dev libglu1-mesa-dev
```

### 2. Unreal Engine 설치 🎮

```bash
git clone -b 4.25 git@github.com:EpicGames/UnrealEngine.git
cd UnrealEngine
./Setup.sh
./GenerateProjectFiles.sh
make
```

### 2. AirSim 리포지토리 클론 및 빌드🔄

```bash
git clone https://github.com/microsoft/AirSim.git
cd AirSim
./setup.sh
./build.sh
```

### 3. Unreal Engine의 Project(Environments) 설정 ⚙️

1. 'AirSim\Unreal\Environments' 에 아래 폴더(Blocks 4.25) 넣기
https://drive.google.com/file/d/1kYiO4VMl9_guA7wXr0a67oRY4MTWn6TO/view?usp=sharing
2. '/home/<username>/Documents/AirSim'에 아래 파일(setting.json) 넣기
https://drive.google.com/file/d/1cTUbTPKsYL4YiGLAqBk7-O0RWVn7LVbc/view?usp=sharing


### 4. Python library 설치 (가상환경 추천)⚙️

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh -O anaconda.sh
bash anaconda.sh
source ~/.bashrc
conda list
conda update conda

conda create -n car python=3.8
conda activate car

pip install -r requirements.txt 
```

가상환경 설정 후 CUDA가 잡히지 않는 다면
```bash
conda install -c anaconda cudatoolkit==[원하는 버전]
ex) conda install -c anaconda cudatoolkit==10.1.243

# CUDA 버전을 못찾는 경우 -c 뒤에 "conda-forge" 옵션 추가
-c conda-forge

# CUDA 먼저 설치하면 아래 커맨드로 어느 정도 상호호환되는 cudnn이 알아서 설치되긴한다.
conda install -c anaconda cudnn

# CUDA(cudatoolkit) version, CUDNN(cudnn) version 확인
conda list
```
### 5. AirSim 실행 🌟

```bash
.UnrealEngine-4.25/Engine/Binaries/Linux/UE4Editor
```
1. 우측 'More' 버튼 클릭
2. 우측 하단 'Browse...' 클릭
3. 3-1에서 추가한 Blocks 4.25 내의 프로젝트 파일을 열기
4. Convert Option이 주어지면 'Convert-In-Place' 선택하기 (선택지가 없으면 좌측 하단의 see more 클릭)
5. 프로젝트가 열리면 하단 에서 'DARL' 더블 클릭
6. 상단 바에서 'Play'버튼 우측에 드롭박스를 열어 'Standalone game' 선택 후 실행


### ※ 차량 위치 변경 방법 🌟
3-2에서 설정한 setting.json의 파일에서 각 차량의 x,y를 변경 후 환경을 재실행

### ⚠️ 주의 사항 ⚠️
1. airsim.CarClient().getCarState("car_name") 을 활용하여 차량의 상태 정보 반환 시 절대좌표가 반환되지 않을 수 있으니, 항상 확인이 필요함
2. airsim.CarClient().simSetObjectPose("car_name", pose, True) 를 활용하여 차량 위치를 변경 시 상대좌표로 이동하니, 절대좌표와 매핑하기 위해선 별도 작업이 필요함

