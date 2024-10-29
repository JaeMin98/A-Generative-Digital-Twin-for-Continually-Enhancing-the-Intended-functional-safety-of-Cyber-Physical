[Cho, Deun-Sol, Jae-Min Cho, and Won-Tae Kim. "A Generative Digital Twin for Continually Enhancing the Intended Functional Safety of Cyber–Physical Systems." IEEE Transactions on Reliability (2024).](https://ieeexplore.ieee.org/abstract/document/10707656)<br><br>

![Title1](https://github.com/user-attachments/assets/22e9f7d5-f4db-45be-baef-2b00a54c4293)<br>
![Title2](https://github.com/user-attachments/assets/450708f0-8edd-4eb8-bf83-a4d2468ab983)<br><br>

# AirSim Installation Guide (Linux) 🛠️

## System Requirements 🖥️

- **OS**: Ubuntu 18.04 / 20.04
- **Hardware**: 8GB RAM, 100GB+ disk space, NVIDIA GPU (CUDA 10.0+)

## Installation Steps 🛠️

### 1. Install Dependency Packages 📦

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libjpeg-dev libpng-dev libtiff-dev libgl1-mesa-dev libglu1-mesa-dev
```

### 2. Install Unreal Engine 🎮

```bash
git clone -b 4.25 git@github.com:EpicGames/UnrealEngine.git
cd UnrealEngine
./Setup.sh
./GenerateProjectFiles.sh
make
```

### 2. Clone and Build AirSim Repository 🔄

```bash
git clone https://github.com/microsoft/AirSim.git
cd AirSim
./setup.sh
./build.sh
```

### 3. Set Up Project (Environments) in Unreal Engine ⚙️

1. Place the folder (Blocks 4.25) in 'AirSim\Unreal\Environments' >>[https://drive.google.com/file/d/1kYiO4VMl9_guA7wXr0a67oRY4MTWn6TO/view?usp=sharing](https://drive.google.com/file/d/1ffDrS2ZHy_ZWV4l-6hOt2p6JpOKS6Ks9/view?usp=sharing)2. Place the following file (settings.json) in '/home/<username>/Documents/AirSim' >>[https://drive.google.com/file/d/1cTUbTPKsYL4YiGLAqBk7-O0RWVn7LVbc/view?usp=sharing](https://drive.google.com/file/d/1OrrDHWXQ8SinTrgtNN7ZApoBkdUr9UcD/view?usp=sharing)

### 4. Install Python Libraries (Virtual Environment Recommended)⚙️

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

If CUDA is not detected after setting up the virtual environment:
```bash
conda install -c anaconda cudatoolkit==[desired version]
ex) conda install -c anaconda cudatoolkit==10.1.243

# Add "conda-forge" after "-c" if the CUDA version cannot be found
-c conda-forge

# Installing CUDA first will automatically install a compatible version of cudnn to some extent
conda install -c anaconda cudnn

# Check versions of CUDA (cudatoolkit) and CUDNN (cudnn)
conda list
```

### 5. Run AirSim 🌟

```bash
.UnrealEngine-4.25/Engine/Binaries/Linux/UE4Editor
```
1. Click the 'More' button on the right
2. Click 'Browse...' at the bottom right
3. Open the project file in the Blocks 4.25 folder added in Step 3-1
4. If the Convert Option is provided, select 'Convert-In-Place' (if not, click "see more" at the bottom left)
5. Once the project is open, double-click 'DARL' at the bottom
6. In the top bar, open the dropdown next to the 'Play' button, select 'Standalone game,' and run

### ※ How to Change Car Position 🌟
Change x and y values for each car in the settings.json file set in Step 3-2, then relaunch the environment.

### ⚠️ Caution ⚠️
1. When returning vehicle status using airsim.CarClient().getCarState("car_name"), absolute coordinates may not be returned; always verify.
2. When changing the vehicle position with airsim.CarClient().simSetObjectPose("car_name", pose, True), it moves with relative coordinates. Additional work is needed to map with absolute coordinates.
