import numpy as np
import tensorflow as tf
from exploration_noise import OUNoise

class Sampler(object):
    def __init__(self,
                 policy,
                 env,
                 batch_size=50,
                 max_step=200,
                 RANDOM_SEED=1234,
                 TAU2=25.0,
                 summary_writer=None):
        self.policy = policy
        self.env = env
        self.batch_size = batch_size
        self.max_step = max_step
        self.RANDOM_SEED = RANDOM_SEED
        self.TAU2 = TAU2
        self.summary_writer = summary_writer

    def add_summary(self, reward):
        global_step = self.policy.session.run(self.policy.global_step)
        summary = tf.Summary()
        summary.value.add(tag="rewards", simple_value=reward)
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()

    def collect_one_episode(self, i):
        state = self.env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        self.noise = OUNoise(self.env.action_space.shape, seed=self.RANDOM_SEED + i)
        epsilon = np.exp(- i / self.TAU2)
        for t in range(self.max_step):
            action = epsilon * self.policy.sampleAction(state[np.newaxis, :]) / self.env.action_space.high
            next_state, reward, done, _ = self.env.step(action)
            # appending the experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            # going to next state
            state = next_state
            if done:
                break
        return dict(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    next_states = next_states,
                    dones = dones
                    )

    def collect_one_batch(self, noise_level):
        episodes = []
        for i_batch in range(self.batch_size):
            episodes.append(self.collect_one_episode(noise_level))
        # prepare input
        states = np.concatenate([episode["states"] for episode in episodes])
        actions = np.concatenate([episode["actions"] for episode in episodes])
        rewards = np.concatenate([episode["rewards"] for episode in episodes])
        next_states = np.concatenate([episode["next_states"] for episode in episodes])
        dones = np.concatenate([episode["dones"] for episode in episodes])
        batch = dict(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    next_states = next_states,
                    dones = dones
                    )
        avg_reward = np.sum(rewards) / float(self.batch_size)
        print("The average reward in the batch is {} per episode".format(avg_reward))
        self.add_summary(avg_reward)
        return batch

    def samples(self, noise_level):
        return self.collect_one_batch(noise_level)
