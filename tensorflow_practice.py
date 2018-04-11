import tensorflow as tf

# Model Parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Input Values
x = tf.placeholder(tf.float32)

# Output of the Model
linear_model = W * x + b

y = tf.placeholder(tf.float32)

# Loss Function
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    File_Writer = tf.summary.FileWriter('graph', sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
        # output = sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
        # print(output)

    print(sess.run([W, b]))
