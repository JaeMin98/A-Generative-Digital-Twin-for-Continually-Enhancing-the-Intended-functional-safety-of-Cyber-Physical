import datetime
import numpy as np
import itertools
import torch
import time

from sac import SAC
from torch.utils.tensorboard import SummaryWriter

import itertools
from replay_memory import ReplayMemory
import config
import environment

parser = config.parser
args = parser.parse_args()

# env
# env = NormalizedActions(gym.make(args.env_name))
env = environment.ENV()

torch.manual_seed(args.seed)
np.random.seed(123456)

# Adversarial_agent
# Adversarial_agent = SAC(env.observation_space_size, env.action_space, args)
Adversarial_agent = SAC(90*5, env.action_space, args)


#Tesnorboard
writer = SummaryWriter('./runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))
model_path = './models/{}_SAC_{}_{}_{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")
# Memory
memory = ReplayMemory(args.replay_size, 123456)

# Training Loop
updates = 0

success_list = []
state= None

for i_episode in itertools.count(1):
    state = np.ones(90*5)

    if args.start_steps > i_episode:
        action = env.action_space.sample()  # Sample random action
    else:
        action = Adversarial_agent.select_action(state)  # Sample action from policy

    if len(memory) > args.batch_size:
        # Number of updates per step in env
        for i in range(args.updates_per_step):
            # Update parameters of all the networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = Adversarial_agent.update_parameters(memory, args.batch_size, updates)

            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
            # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
            # writer.add_scalar('loss/policy', policy_loss, updates)
            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
            writer.add_scalar('entropy_temprature/alpha', alpha, updates)
            updates += 1

    next_state, reward, done, success = env.step(action) # Step
    next_state = list(itertools.chain(*next_state))
    done, success = False, False

    memory.push(state, action, reward, next_state, done)

    #success rate 계산
    success_list.append(success)
    average_num = 20
    n = len(success_list)

    if n >= average_num: sum_last = sum(success_list[n-average_num:n])
    else: sum_last = sum(success_list)

    success_rate = 100*(sum_last/average_num)
    
    writer.add_scalar('score/train', reward, i_episode)
    writer.add_scalar('success_rate/train', success_rate, i_episode)
    
    if (env.collision_info_1 == True) and (env.collision_info_2 == True):
        Adversarial_agent.save_model(model_path + str(i_episode)+".tar")
        env.write_figure_data(model_path + str(i_episode)+".csv")

    print("Episode: {}, reward: {}, actions: [{}, {}, {}, {}, {}, {}]".format(i_episode, round(reward, 2), action[0], action[1], action[2], action[3], action[4], action[5]))

env.reset()