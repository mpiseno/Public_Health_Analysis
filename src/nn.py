from sklearn.model_selection import train_test_split

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data extraction
data = pd.read_csv('at_risk_queer.csv', sep=',')
X = data.values[:, 0:(data.shape[1] - 2)]
Y = data.values[:, data.shape[1] - 1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# Make One-Hot Vectors
temp0 = np.zeros([Y_train.shape[0], 1])
temp1 = np.zeros([Y_train.shape[0], 1])
mask0 = Y_train == 1.0
mask1 = Y_train == 0.0
temp0[mask0] = 1
temp1[mask1] = 1
Y_train = np.concatenate((temp0, temp1), axis=1)

temp0 = np.zeros([Y_test.shape[0], 1])
temp1 = np.zeros([Y_test.shape[0], 1])
mask0 = Y_test == 1.0
mask1 = Y_test == 0.0
temp0[mask0] = 1
temp1[mask1] = 1
Y_test = np.concatenate((temp0, temp1), axis=1)

# Network Hyper-parameters
learning_rate = 0.00005
epochs = 1000
num_examples = X.shape[0]

# Placeholders for data
inputs = tf.placeholder(dtype=tf.float32, shape=[None, X.shape[1]])
labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])

# Define model architecture
dense1 = tf.layers.dense(inputs=inputs, units=20, kernel_initializer=tf.initializers.random_normal,
                         bias_initializer=tf.initializers.ones, activation=tf.nn.relu)
dense2 = tf.layers.dense(inputs=dense1, units=20, kernel_initializer=tf.initializers.random_normal,
                         bias_initializer=tf.initializers.ones, activation=tf.nn.relu)
dense3 = tf.layers.dense(inputs=dense2, units=2, kernel_initializer=tf.initializers.random_normal,
                         bias_initializer=tf.initializers.ones, activation=tf.nn.relu)
logits = tf.nn.softmax(dense3, axis=1)

# Logistics
def get_cost(predicted, actual):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted, labels=actual))

def get_acc(predicted, actual):
    isEqual = tf.equal(tf.argmax(predicted, axis=1), tf.argmax(actual, axis=1))
    return 100 * tf.reduce_mean(tf.cast(isEqual, tf.float32))

cost = get_cost(logits, labels)
accuracy = get_acc(logits, labels)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)
saver = tf.train.Saver()

with tf.Session() as sess:
    print("Training...")
    sess.run(tf.global_variables_initializer())

    total_cost = []
    total_accuracy = []
    # For each epoch we feed the network ALL training examples, in batches
    for epoch in range(epochs):
        epoch_cost = 0
        epoch_acc = 0

        # Evaluate the loss, accuracy, and optimizer in that order, feeding in the neccesary values
        loss, acc, opt = sess.run([cost, accuracy, train], feed_dict={inputs: X_train, labels: Y_train})

        # Add the loss and accuracy for this batch
        epoch_cost += loss
        epoch_acc += acc
        total_cost.append(epoch_cost)
        total_accuracy.append(epoch_acc)

        # Periodically print out training progress
        if epoch % 50 == 0:
            print("Epoch: {:.3f}, cost: {:.3f}, training accuracy: {:.3f}%".format(epoch, epoch_cost, epoch_acc))

    # Now that model is trained, run test data through it
    print("Training done. Validating the model...\n")
    test_cost, test_acc = sess.run([cost, accuracy], feed_dict={inputs: X_test, labels: Y_test})
    print("Test cost: {:.3f}, test accuracy: {:.3f}\n".format(test_cost, test_acc))

    # Save model and plot accuracy vs epochs
    print("Saving model...")
    save_path = saver.save(sess, "/tmp/model.ckpt")
    plt.plot(np.arange(epochs), np.array(total_accuracy))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
'''
training_data_x = pd.read_excel("Data.xlsx")
X_train = training_data_x.as_matrix()
print(X_train.shape)


sess = tf.Session();
with sess.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(60,9))
    y = tf.placeholder(dtype=tf.float32)
    y_ = tf.placeholder(dtype=tf.float32)

    dense1 = tf.layers.dense(inputs=x, units=12, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=6, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=3, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense3, rate=0.2, training=True)
    logits = tf.layers.dense(inputs=dropout, units=1)

    error = tf.abs(logits - y)
    mean = tf.reduce_mean(error)
    train_op1 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(error)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("training...")
    for _ in range(50):
        sess.run(train_op1, {x: X_train[0:60,0:9], y: X_train[0:60,9]})
        print(sess.run(mean, {x: X_train[61:121,0:9], y: X_train[61:121,9]}))

        print(sess.run(logits, {x: })

'''
