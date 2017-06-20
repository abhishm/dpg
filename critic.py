import random
import numpy as np
import tensorflow as tf

class DQNAgent(object):
    def __init__(self, session,
                       optimizer,
                       q_network,
                       observation_space,
                       action_space,
                       discount,
                       target_update_rate,
                       q_error_threshold,
                       summary_writer=None,
                       summary_every=100):

        # tensorflow machinery
        self.session        = session
        self.optimizer      = optimizer
        self.summary_writer = summary_writer
        self.summary_every  = summary_every
        self.no_op          = tf.no_op()

        # model components
        self.q_network     = q_network

        # Q learning parameters
        self.observation_space  = observation_space
        self.action_space       = action_space
        self.discount           = discount
        self.target_update_rate = target_update_rate
        self.q_error_threshold  = q_error_threshold

        # counters
        self.train_itr = 0

        # create and initialize variables
        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        if self.summary_writer is not None:
            self.summary_writer.add_graph(self.session.graph)
            self.summary_every = summary_every

    def create_input_placeholders(self):
        with tf.name_scope("inputs"):
            self.states = tf.placeholder(tf.float32, (None,) + self.observation_space, "states")
            self.actions = tf.placeholder(tf.float32, (None,) + self.action_space, "actions")
            self.rewards = tf.placeholder(tf.float32, (None,), "rewards")
            self.next_states = tf.placeholder(tf.float32, (None,) + self.observation_space, "next_states")
            self.next_actions = tf.placeholder(tf.float32, (None,) + self.action_space, "next_actions")
            self.dones = tf.placeholder(tf.bool, (None,), "dones")

    def create_variables_for_q_values(self):
        with tf.name_scope("action_values"):
            with tf.variable_scope("q_network"):
                self.q_values = self.q_network(self.states,
                                               self.actions,
                                               self.observation_space,
                                               self.action_space)
                self.q_gradients = tf.gradients(self.q_values, self.actions)

    def create_variables_for_target(self):
        with tf.name_scope("target_values"):
            not_the_end_of_an_episode = 1.0 - tf.cast(self.dones, tf.float32)
            with tf.variable_scope("target_network"):
                self.target_q_values = self.q_network(self.next_states,
                                                      self.next_actions,
                                                      self.observation_space,
                                                      self.action_space)

            self.target_q_values = tf.multiply(self.target_q_values, not_the_end_of_an_episode)
            self.target_values = self.rewards + self.discount * self.target_q_values

    def create_variables_for_optimization(self):
        with tf.name_scope("optimization"):
            square_diff = 0.5 * tf.square(self.q_values - self.target_values)
            self.loss = tf.reduce_mean(square_diff)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_variables)
            self.clipped_gradients = [(tf.clip_by_norm(grad, 40), var) for grad, var in self.gradients]
            self.train_op = self.optimizer.apply_gradients(self.clipped_gradients)
            self.var_norm = tf.global_norm(self.trainable_variables)
            self.grad_norm = tf.global_norm([grad for grad, var in self.gradients])

    def create_variables_for_target_network_update(self):
        with tf.name_scope("target_network_update"):
            target_ops = []
            q_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network")
            target_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network")
            for v_source, v_target in zip(q_network_variables, target_network_variables):
                target_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                target_ops.append(target_op)
            self.target_update = tf.group(*target_ops)

    def create_summaries(self):
        self.loss_summary = tf.summary.scalar("q/loss", self.loss)
        self.var_norm_summary = tf.summary.scalar("q/var_norm", self.var_norm)
        self.grad_norm_summary = tf.summary.scalar("q/grad_norm", self.grad_norm)

    def merge_summaries(self):
        self.summarize = tf.summary.merge([self.loss_summary,
                                           self.var_norm_summary,
                                           self.grad_norm_summary])

    def create_variables(self):
        self.create_input_placeholders()
        self.create_variables_for_q_values()
        self.create_variables_for_target()
        self.create_variables_for_optimization()
        self.create_variables_for_target_network_update()
        self.create_summaries()
        self.merge_summaries()

    def compute_q_gradients(self, states, actions):
        return self.session.run(self.q_gradients, {self.states: states,
                                                   self.actions: actions})[0]

    def update_parameters(self, batch):
        write_summary = self.train_itr % self.summary_every == 0
        _, summary = self.session.run([self.train_op,
                                       self.summarize if write_summary else self.no_op],
                                      {self.states: batch["states"],
                                       self.actions: batch["actions"],
                                       self.rewards: batch["rewards"],
                                       self.next_states: batch['next_states'],
                                       self.next_actions: batch["next_actions"],
                                       self.dones: batch["dones"]})

        self.session.run(self.target_update)

        if write_summary:
            self.summary_writer.add_summary(summary, self.train_itr)

        self.train_itr += 1
