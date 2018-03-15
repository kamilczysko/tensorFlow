import tensorflow as tf
import prep_data_loader as data
import numpy as np
from PIL import Image

print('loading data...')
data_input = data.learn_data_array
data_target = data.learn_target_array
print('loaded!')
print('placeholders...')
x = tf.placeholder(tf.float32, [None, 784], name='xPlaceholder')
y = tf.placeholder(tf.float32, [None, 6],name='yPlaceholder')
print('done')
learning_rate = 0.5
batch_size = 10
epochs = 55

hidden_nodes_1 = 64
hidden_nodes_2 = 10
output_nodes = 6

predict_op = []
test_data = data.get_data_to_test(2)

def network(data):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, hidden_nodes_1])),
                      'biases': tf.Variable(tf.random_normal([hidden_nodes_1]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([hidden_nodes_1, output_nodes])),
                    'biases': tf.Variable(tf.random_normal([output_nodes]))}

    out_l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    out_l1 = tf.nn.relu(out_l1)

    global predict_op
    out = tf.nn.softmax(tf.add(tf.matmul(out_l1, output_layer['weights']), output_layer['biases']))
    predict_op = tf.argmax(out, 1, name='w3')
    return out


def train(x):
    print("training......")
    guess = network(x)
    # print(guess)
    y_clipped = tf.clip_by_value(guess, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    init_operations = tf.global_variables_initializer()

    correct_guesses = tf.equal(tf.argmax(y, 1), tf.argmax(guess, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_guesses, tf.float32))

    saver = tf.train.Saver();
    global predict_op
    with tf.Session() as sess:
        sess.run(init_operations)
        total_batch = int(len(data_input) / 50)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = data.get_batch()
                _, c = sess.run([optimizer, cross_entropy],
                                feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        print(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
        print(sess.run(predict_op, feed_dict={x: np.reshape(test_data[0], newshape=(1, 784))}))

        saver.save(sess, 'networks/myNetwork')

train(x)
print(predict_op)
test_data = data.get_data_to_test(2)
print(data.answer_array)

with tf.Session() as sess:
    tf.global_variables_initializer()
    saver = tf.train.import_meta_graph('networks/myNetwork.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./networks'))
    inp = np.reshape(test_data[0], newshape=(1, 784))
    graph = tf.get_default_graph()
    pred = graph.get_tensor_by_name('w3:0')
    x1 = graph.get_tensor_by_name('xPlaceholder:0')
    print(sess.run(pred ,feed_dict={x1: inp}))

img = np.reshape(test_data[0] * 255, newshape=(28, 28))
pic = Image.fromarray(img)
pic.show()

