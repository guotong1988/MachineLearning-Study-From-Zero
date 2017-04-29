# input reward to train neural network to get right action 
import tensorflow as tf

state_input = tf.placeholder(tf.float32, [None, 3])
action_input = tf.placeholder(tf.float32, [None, 2])
reward_input = tf.placeholder(tf.float32, [None])

input_data = [[1,0,1],[1,1,0],[0,1,1],
              [1,0,0],[0,1,0],[0,0,1]]

action_data = [[0,1],[0,1],[0,1],
               [1,0],[1,0],[1,0]]
reward_data = [1,1,1,1,1,1]

fake_action_data =[[1,0],[1,0],[1,0],
                   [0,1],[0,1],[0,1]]
fake_reward_data = [0,0,0,0,0,0]

W1 = tf.Variable(tf.random_normal([3, 30]))
b1 = tf.Variable(tf.random_normal([30]))

W2 = tf.Variable(tf.random_normal([30, 2]))
b2 = tf.Variable(tf.random_normal([2]))

temp1 = tf.add(tf.matmul(state_input,W1),b1)
temp1 = tf.nn.relu(temp1)
predict_action = tf.add(tf.matmul(temp1,W2),b2)

selected_q_value = tf.multiply(predict_action, action_input)
selected_q_value = tf.reduce_sum(selected_q_value, axis=1)
cost = tf.reduce_mean(tf.square(reward_input - selected_q_value))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(0,5000):
        _,cost_print1 = sess.run([optimizer,cost],feed_dict={
            state_input:input_data,
            action_input:action_data,
            reward_input:reward_data
        })
        _,cost_print2 = sess.run([optimizer, cost], feed_dict={
            state_input: input_data,
            action_input: fake_action_data,
            reward_input: fake_reward_data
        })
        print(cost_print1)
        print(cost_print2)
        print("-------")
    print(predict_action.eval(feed_dict={state_input: input_data}))
