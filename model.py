import tensorflow as tf

def policy_network(states, observation_space, action_space, action_bound):
   """ define policy neural network """
   W1 = tf.get_variable("W1", observation_space + (400,),
                        initializer=tf.contrib.layers.xavier_initializer())
   b1 = tf.get_variable("b1", (400,),
                        initializer=tf.constant_initializer(0))
   h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
   W2 = tf.get_variable("W2", (400, 300),
                        initializer=tf.contrib.layers.xavier_initializer())
   b2 = tf.get_variable("b2", (300,),
                        initializer=tf.constant_initializer(0))
   h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
   W3 = tf.get_variable("W3", (300,) + action_space,
                        initializer=tf.contrib.layers.xavier_initializer())
   b3 = tf.get_variable("b3", action_space,
                        initializer=tf.constant_initializer(0))
   a = tf.nn.tanh(tf.matmul(h2, W3) + b3)
   a = tf.multiply(a, action_bound)
   return a


def critic_network(states, actions, observation_space, action_space):
    W1 = tf.get_variable("W1", observation_space + (400,),
                                initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", (400,),
                                initializer=tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
    h1 = tf.concat([h1, actions], axis=1)
    W2 = tf.get_variable("W2", (400 + action_space[0], 300),
                                initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", (300,),
                                initializer=tf.constant_initializer(0))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    W3 = tf.get_variable("W3", (300, 1),
                                initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", (1,),
                                initializer=tf.constant_initializer(0))
    q = tf.matmul(h2, W3) + b3
    return tf.reshape(q, (-1,))
