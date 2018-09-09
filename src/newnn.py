#LETS GOOOOOO
import tensorflow as tf;
import pandas as pd;
import numpy as np
training_data_x = pd.read_excel("QueerByState.xlsx")
X_train = training_data_x.as_matrix()
print(X_train.shape)


y = tf.placeholder(dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32, shape=(None, 3))

w = tf.Variable(tf.zeros([3,1]))
b = tf.Variable(tf.zeros([1]))

y_ = tf.matmul(x,w) + b;

cost = tf.abs(y-y_)
train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        for _ in range(100):
            sess.run(train_op, {x: X_train[0:40,1:4], y: X_train[0:40,0]})
        print(np.average(sess.run(cost, {x: X_train[0:40,1:4], y: X_train[0:40,0]})))
    print(np.average(sess.run(cost, {x: X_train[40:49,1:4], y: X_train[40:49,0]})))