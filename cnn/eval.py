#-*- coding:utf-8 -*-
from input_data import data_load
from inference import inference_net
import tensorflow as tf
import os

data_input = "test.data"
label,ceshi = data_load(data_input)
print label
def evaluate(data_input):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[3,16,16,60],name = 'x-input')
        y_ = tf.placeholder(tf.float32,[3,3],name = 'y-input')
        y = inference_net(x,False,None)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        variable_averages = tf.train.ExponentialMovingAverage(0.99)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("save_model/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                jieguo = sess.run([y,cross_entropy_mean],feed_dict = {x:ceshi,y_:label})
                print "The results is:"
                print jieguo
            else:
                print ("No checkpoint file found")

def main(argv=None):
    evaluate(data_input)

if __name__ == "__main__":
    tf.app.run()
