#-*- coding:utf-8 -*-
from input_data import data_load
from inference import inference_net
import tensorflow as tf
import os
data_file = "train.raw.data"

BATCH_SIZE = 100
def train(data_file):
    print ("Loading Data ...")
    xs,ys = data_load(data_file)
    print ("Finished data load")
    x = tf.placeholder(tf.float32,[BATCH_SIZE,16,16,60],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[BATCH_SIZE,3],name = 'y-input')

    regularizer = tf.contrib.layers.l2_regularizer(0.0001)

    y = inference_net(x,True,regularizer)

    global_step = tf.Variable(0,trainable = False)

    variable_averages = tf.train.ExponentialMovingAverage(0.99,global_step)

    main_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("loss"))
    learning_rate = tf.train.exponential_decay(0.8,global_step,len(ys)/BATCH_SIZE,0.9) 
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

    with tf.control_dependencies([train_step,main_op]):
        train_op = tf.no_op(name = 'train')

    #保存模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range((len(ys)-BATCH_SIZE)/2):
            xs_tf = xs[i*2:i*2+BATCH_SIZE]
            ys_tf = ys[i*2:i*2+BATCH_SIZE]
                
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict = {x:ys_tf,y_:xs_tf})
            if i % 200 == 0:
                print ("After %d training steps,loss on training batch is %g." %(step,loss_value))
                saver.save(sess,"save_model/train_oyp.ckpt",global_step = global_step)

def main(argv=None):
    train(data_file)

if __name__ == "__main__":
    tf.app.run()

