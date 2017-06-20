import tensorflow as tf
import numpy as np
import json
import gym
from tqdm import trange
import matplotlib.pyplot as plt
from critic import DQNAgent
from actor import DeterministicPolicy
from replay_buffer import ReplayBuffer
from model import policy_network
from model import critic_network
from sampler import Sampler

config = json.load(open("configuration.json"))

# Environment parameters
env_name = config["env_name"]
env = gym.make(env_name)
observation_space = env.observation_space.shape
action_space = env.action_space.shape
action_bound = env.action_space.high[0]

# Policy nework parameters
policy_session = tf.Session()
policy_optimizer = tf.train.AdamOptimizer(learning_rate=config["policy_learning_rate"])
policy_writer = tf.summary.FileWriter("policy/")
policy_summary_every = 10

policy = DeterministicPolicy(policy_session,
                             policy_optimizer,
                             policy_network,
                             observation_space,
                             action_space,
                             action_bound,
                             config["max_gradient"],
                             config["target_update_rate"],
                             summary_writer=policy_writer,
                             summary_every=policy_summary_every)

# Initializing Sampler
sampler = Sampler(policy,
                  env,
                  config["batch_size"],
                  config["max_step"],
                  summary_writer=policy_writer)
#
# Q-network parameters
q_session = tf.Session()
q_optimizer = tf.train.AdamOptimizer(config["q_learning_rate"])
q_writer = tf.summary.FileWriter("q/")
q_summary_every = 10

dqn_agent = DQNAgent(q_session,
                     q_optimizer,
                     critic_network,
                     observation_space,
                     action_space,
                     config["discount"],
                     config["target_update_rate"],
                     config["q_error_threshold"],
                     summary_writer=q_writer,
                     summary_every=q_summary_every)

# Initializing ReplayBuffer
buffer_size = config["buffer_size"]
sample_size = config["sample_size"]
q_network_updates = config["q_network_updates"]
replay_buffer = ReplayBuffer(buffer_size)
#
# # Training

def update_q_parameters(batch):
    batch["next_actions"] = policy.compute_next_actions(batch["next_states"])
    dqn_agent.update_parameters(batch)




for i in trange(1):
    batch = sampler.samples(i)
    replay_buffer.add_batch(batch)
    if sample_size <= replay_buffer.num_items:
        random_batch = replay_buffer.sample_batch(sample_size)
        update_q_parameters(random_batch)
        action_grads = dqn_agent.compute_q_gradients(random_batch["states"],
                                                     random_batch["actions"])

        policy.update_parameters(random_batch["states"], action_grads)
