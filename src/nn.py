#LETS GOOOOOO
import tensorflow as tf;
import pandas as pd;
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


