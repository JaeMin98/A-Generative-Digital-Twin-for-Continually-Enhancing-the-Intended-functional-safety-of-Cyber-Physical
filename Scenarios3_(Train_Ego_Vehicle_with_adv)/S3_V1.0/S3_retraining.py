import datetime
import numpy as np
import itertools
import torch
import time
import os
import shutil

from sac import SAC
from torch.utils.tensorboard import SummaryWriter

from replay_memory import ReplayMemory
import config
import environment

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"폴더 생성: {folder_path}")
    else:
        print(f"폴더가 이미 존재함: {folder_path}")

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

# Agent
agent = SAC(env.observation_space_size, env.action_space, args)
agent.load_checkpoint("ego_agent_model/ego_agent.tar")

# Adv_agent
Adv_args = args
Adv_args.hidden_size = 64

Adv_agent = SAC(env.observation_space_size_adv, env.action_space_adv, Adv_args)
Adv_agent.load_checkpoint("Adv_agent_model/S2_1/S2_2_1.tar")

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))
model_path = 'models/{}_SAC_{}_{}_{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")

log_path = 'training_logs/{}_SAC_{}_{}_{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")

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

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    adv_state = env.get_Adv_state()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in env
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                append_to_file(os.path.join(log_path,"critic_1_loss"), critic_1_loss)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                append_to_file(os.path.join(log_path,"critic_2_loss"),critic_2_loss)
                writer.add_scalar('loss/policy', policy_loss, updates)
                append_to_file(os.path.join(log_path,"policy_loss"),policy_loss)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                append_to_file(os.path.join(log_path,"ent_loss"),ent_loss)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                append_to_file(os.path.join(log_path,"alpha"),alpha)
                updates += 1

        env.step(action) # Step
        adv_state = env.get_Adv_state()
        Adv_action = Adv_agent.select_action(adv_state)
        # print(adv_state,Adv_action)
        env.step_for_Adv(Adv_action)
        
        if total_numsteps%100 == 0 :
            env.set_car_control_of_front_random()

        time.sleep(0.03)
        
        next_state, reward, done, success, distance_X = env.observation()
        

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        if(episode_steps == env._max_episode_steps):
            done = False
            success = True
            reward += 250

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    #success rate 계산
    
    success_list.append(success)
    average_num = 20
    n = len(success_list)

    if n >= average_num: sum_last = sum(success_list[n-average_num:n])
    else: sum_last = sum(success_list)

    success_rate = 100*(sum_last/average_num)

    if total_numsteps > args.num_steps:
        break
    
    append_to_file(os.path.join(MDP_log_path,"success"),success)
    append_to_file(os.path.join(MDP_log_path,"episode_reward"),episode_reward)
    append_to_file(os.path.join(MDP_log_path,"success_rate"),success_rate)

    writer.add_scalar('score/train', episode_reward, i_episode)
    writer.add_scalar('success_rate/train', success_rate, i_episode)
    writer.add_scalar('distance/train', distance_X, i_episode)
    

    agent.save_model(model_path + str(i_episode)+".tar")

    training_log = "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2))
    print(training_log)
    append_to_file(os.path.join(log_path,"training_log"),training_log)
    append_to_file(os.path.join(log_path,"distance"),distance_X)

env.reset()