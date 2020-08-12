import tensorflow as tf
import os
import sys
import numpy as np
from conf import config
sys.path.append("./conf")
from data_utils import Data
if __name__ == '__main__':
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./runs/1597229589/checkpoints/model-780.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./runs/1597229589/checkpoints/'))

    test_data = Data(data_source=config.test_data_source,
                          alphabet=config.alphabet,
                          l0=config.l0,
                          batch_size=config.batch_size,
                          no_of_classes=config.no_of_classes)
    test_data.load_Test_Data()
    graph = tf.get_default_graph()
    x_ = graph.get_tensor_by_name("Input-Layer/input_x:0")
    keep_prob_ = graph.get_tensor_by_name("Input-Layer/dropout_keep_prob:0")
    pred_test = graph.get_tensor_by_name("OutputLayer/scores:0")
    xin = test_data.test_get()
    feed_dict = {x_: xin, keep_prob_: 1.0}
    pred = sess.run([pred_test], feed_dict)
    label = np.argmax(pred[0], axis=1)
    print(label)
