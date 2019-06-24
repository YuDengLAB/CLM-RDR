import tensorflow as tf
import os
import sys
import numpy as np
sys.path.append("./conf")
import config
from data_utils import Data
from char_cnn import CharConvNet
if __name__ == '__main__':
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./runs/1559730669/checkpoints/model-16400.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./runs/1559730669/checkpoints/'))

    test_data = Data(data_source=config.test_data_source,
                          alphabet=config.alphabet,
                          l0=config.l0,
                          batch_size=config.batch_size,
                          no_of_classes=config.no_of_classes)
    test_data.load_Test_Data()

    graph = tf.get_default_graph()
    # for op in graph.get_operations():
    #     print(op.name)
    x_ = graph.get_tensor_by_name("Input-Layer/input_x:0")
    keep_prob_ = graph.get_tensor_by_name("Input-Layer/dropout_keep_prob:0")
    pred_test = graph.get_tensor_by_name("OutputLayer/score:0")
    xin = test_data.test_get()
    feed_dict = {x_: xin, keep_prob_: 1.0}
    pred = sess.run([pred_test], feed_dict)
    pred_ = []
    for i in pred:
        index = np.argmax(i)
        pred_.append(index)
    print(pred_)
