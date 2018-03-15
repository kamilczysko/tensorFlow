import tensorflow as tf
import prep_data_loader as data
import numpy as np

print(data.answer_array)
dat = data.answer_array

def guess(data):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('networks/myNetwork.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./networks'))
        inp = np.reshape(data, newshape=(1, 784))
        graph = tf.get_default_graph()
        pred = graph.get_tensor_by_name('w3:0')
        x = graph.get_tensor_by_name('xPlaceholder:0')
        result = sess.run(pred ,feed_dict={x: inp})
        for i in dat:
            if result == i[1]:
                return i[0]
