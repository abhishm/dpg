# import random
import numpy as np
import tensorflow as tf

class DeterministicPolicy(object):

  def __init__(self, session,
                     optimizer,
                     policy_network,
                     observation_space,
                     action_space,
                     action_bound,
                     max_gradient,
                     target_update_rate,
                     summary_writer=None,
                     summary_every=100):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer
    self.summary_every  = summary_every
    self.no_op           = tf.no_op()

    # model components
    self.policy_network = policy_network
    self.observation_space = observation_space
    self.action_space = action_space
    self.action_bound = action_bound

    # training parameters
    self.max_gradient = max_gradient
    self.target_update_rate = target_update_rate

    #counter
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    # create and initialize variables
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = summary_every

  def create_input_placeholders(self):
    with tf.name_scope("inputs"):
      self.states = tf.placeholder(tf.float32, (None,) + self.observation_space,
                                                                name="states")
      self.q_gradients = tf.placeholder(tf.float32, (None,) + self.action_space,
                                                                name="q_gradients")

  def create_variables_for_actions(self):
    with tf.variable_scope("policy_network"):
      self.action = self.policy_network(self.states,
                                      self.observation_space,
                                      self.action_space,
                                      self.action_bound)
    with tf.variable_scope("clone_network"):
      self.clone_action = self.policy_network(self.states,
                                      self.observation_space,
                                      self.action_space,
                                      self.action_bound)

  def create_variables_for_optimization(self):
    with tf.name_scope("optimization"):
      self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope="policy_network")
      self.gradients = tf.gradients(self.action, self.trainable_variables,
                                        grad_ys=-self.q_gradients)
      self.train_op = self.optimizer.apply_gradients(zip(self.gradients, self.trainable_variables))
      self.grad_norm = tf.global_norm(self.gradients)
      self.var_norm = tf.global_norm(self.trainable_variables)

  def update_clone(self):
    self.clone_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="clone_network")
    target_ops = []
    for u, v in zip(self.trainable_variables, self.clone_variables):
        target_op = v.assign_sub(self.target_update_rate * (v - u))
        target_ops.append(target_op)
    self.update_clone = tf.group(*target_ops)

  def create_summaries(self):
    self.grad_norm_summary = tf.summary.scalar("policy/grad_norm", self.grad_norm)
    self.var_norm_summary = tf.summary.scalar("policy/var_norm", self.var_norm)

  def merge_summaries(self):
    self.summarize = tf.summary.merge([self.grad_norm_summary,
                                       self.var_norm_summary])

  def create_variables(self):
    self.create_input_placeholders()
    self.create_variables_for_actions()
    self.create_variables_for_optimization()
    self.update_clone()
    self.create_summaries()
    self.merge_summaries()

  def sampleAction(self, states):
    a = self.session.run(self.action, {self.states: states})[0]
    return a

  def compute_next_actions(self, next_states):
    return self.session.run(self.clone_action, {self.states: next_states})


  def update_parameters(self, states, q_gradients):
    write_summary = self.train_itr % self.summary_every == 0
    _, summary = self.session.run([self.train_op,
                                   self.summarize if write_summary else self.no_op],
                                  {self.states: states,
                                   self.q_gradients: q_gradients})
    self.session.run(self.update_clone)
    if write_summary:
        self.summary_writer.add_summary(summary, self.train_itr)

  @property
  def train_itr(self):
    return self.session.run(self.global_step)
