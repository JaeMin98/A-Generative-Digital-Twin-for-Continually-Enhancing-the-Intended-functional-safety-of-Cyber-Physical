import datetime
import numpy as np
import itertools
import torch
import time
import os
import wandb

from sac import SAC
from replay_memory import ReplayMemory
import config
import environment
import shutil

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"폴더 생성: {folder_path}")
    else:
        pass

def append_to_file(file_path, text):
    file_path += '.txt'
    text = str(text)
    with open(file_path, 'a') as file:
        file.write(text + "\n")

def copy_file_to_folder(src_file_path, dest_folder_path):
    shutil.copy(src_file_path, dest_folder_path)

parser = config.parser
args = parser.parse_args()

# env
# env = NormalizedActions(gym.make(args.env_name))
env = environment.ENV()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Adversarial_agent
Adversarial_agent = SAC(env.observation_space_size, env.action_space, args)

# Ego_agent
Ego_args = args
Ego_args.hidden_size = 128

Ego_agent = SAC(env.observation_space_size_of_ego, env.action_space_of_ego, Ego_args)
Ego_agent.load_checkpoint("ego_agent_model/ego_agent.tar")

now = datetime.datetime.now()
date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(
    project="Car_case4",
    config=args,
    name="right_behind__"+date_time_str  # 원하는 run 이름 지정
)
model_path = 'models/{}_SAC_{}_{}_{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")
log_path = 'training_logs/{}_SAC_{}_{}_{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")

# writer = SummaryWriter('/home/smartcps/Desktop/Airsim_model_files/Scenario_2_runs/Version2.0_{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                              args.policy, "autotune" if args.automatic_entropy_tuning else ""))
# model_path = '/home/smartcps/Desktop/Airsim_model_files/Scenario_2_models/Version2.0_{}_SAC_{}_{}_{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                              args.policy, "autotune" if args.automatic_entropy_tuning else "")

log_folder_path = "./training_logs"
MDP_log_path = os.path.join(log_path,"MDP_log")
create_folder_if_not_exists(log_folder_path)
create_folder_if_not_exists(log_path)
create_folder_if_not_exists(MDP_log_path)
copy_file_to_folder("./config.py", log_path)


# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

success_list = []
ROI_success_list = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_reward_Ego = 0
    episode_steps = 0
    done = False
    state = env.reset()
    IS_episode_ROI_coll = False

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = Adversarial_agent.select_action(state)  # Sample action from policy

        Ego_state = env.get_Ego_state()
        Ego_action = Ego_agent.select_action(Ego_state)

        if len(memory) > args.batch_size:
            # Number of updates per step in env
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = Adversarial_agent.update_parameters(memory, args.batch_size, updates)
                updates += 1
            wandb.log({"Network/critic_1": critic_1_loss}, step=i_episode)
            wandb.log({"Network/critic_2": critic_2_loss}, step=i_episode)
            wandb.log({"Network/policy": policy_loss}, step=i_episode)
            wandb.log({"Network/entropy_loss": ent_loss}, step=i_episode)
            wandb.log({"Network/alpha": alpha}, step=i_episode)
            append_to_file(os.path.join(log_path,"critic_1_loss"), critic_1_loss)
            append_to_file(os.path.join(log_path,"critic_2_loss"),critic_2_loss)
            append_to_file(os.path.join(log_path,"policy_loss"),policy_loss)
            append_to_file(os.path.join(log_path,"ent_loss"),ent_loss)
            append_to_file(os.path.join(log_path,"alpha"),alpha)
            
        env.step(action) # Step
        env.step_for_Ego(Ego_action)
        
        if total_numsteps%100 == 0 :
            env.set_car_control_of_front_random()

        time.sleep(0.03)
        
        next_state, reward, done, success, Ego_reward = env.observation()
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        episode_reward_Ego += Ego_reward

        if(success) : IS_episode_ROI_coll = True

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        if(episode_steps == 300) : done = True
    
    #success rate 계산
    #------success rate 계산-------
    success_list.append(env.IsCollision)
    average_num = 20
    n = len(success_list)

    if n >= average_num: sum_last = sum(success_list[n-average_num:n])
    else: sum_last = sum(success_list)

    success_rate = 100*(sum_last/average_num)
    #-----------------------------------

    #------ROI_success rate 계산-------
    ROI_success_list.append(IS_episode_ROI_coll)
    ROI_average_num = 20
    ROI_n = len(ROI_success_list)

    if n >= ROI_average_num: ROI_sum_last = sum(ROI_success_list[ROI_n-ROI_average_num:n])
    else: ROI_sum_last = sum(ROI_success_list)

    ROI_success_rate = 100*(ROI_sum_last/ROI_average_num)
    #-----------------------------------

    if total_numsteps > args.num_steps:
        break
    
    wandb.log({"episode/score": episode_reward}, step=i_episode)
    wandb.log({"episode/success_rate": success_rate}, step=i_episode)
    wandb.log({"episode/ROI_success_rate": ROI_success_rate}, step=i_episode)

    if (env.IsCollision == True):
        collision_episode_path = model_path+'collision'
        create_folder_if_not_exists(collision_episode_path)
        env.write_figure_data(collision_episode_path + '/' + str(i_episode)+".csv")

        if(IS_episode_ROI_coll == True):
            ROI_collision_episode_path = model_path+'ROI_collision'
            create_folder_if_not_exists(ROI_collision_episode_path)
            env.write_figure_data(ROI_collision_episode_path + '/' + str(i_episode)+".csv")
    else :
        fail_episode_path = model_path+'fail'
        create_folder_if_not_exists(fail_episode_path)
        env.write_figure_data(fail_episode_path + '/' + str(i_episode)+".csv")

    training_log = "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2))
    print(training_log)
    append_to_file(os.path.join(log_path,"training_log"),training_log)

    if(success_rate == 100):
        break
    
env.reset()