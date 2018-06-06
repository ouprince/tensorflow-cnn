# -*- coding:utf-8 -*-
import tensorflow as tf
def inference_net(input_tensor,train,regularizer):
    with tf.variable_scope("layer1_conv1"):
        conv1_weights = tf.get_variable("weights",[5,5,60,10],initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv1_biass = tf.get_variable("biasses",[10],initializer = tf.constant_initializer(0.01))

        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides = [1,1,1,1],padding = 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biass))
    
    with tf.name_scope("layer2_pool1"):
        pool1 = tf.nn.max_pool(relu1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
    
    with tf.variable_scope("layer3_conv2"):
        conv2_weights = tf.get_variable("weights",[3,3,10,1],initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv2_biass = tf.get_variable("biasses",[1],initializer = tf.constant_initializer(0.01))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides = [1,1,1,1],padding = 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biass))

    with tf.name_scope("layer4_pool2"):
        pool2 = tf.nn.max_pool(relu2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
    
    pool_shape = pool2.get_shape().as_list() #[None,4,4,100]
    nodes_pool = pool_shape[1] * pool_shape[2] * pool_shape[3]
    print pool_shape
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes_pool])

    #进入全连接层
    with tf.variable_scope("layer5_fc1"):
        fc1_weights = tf.get_variable("weights",[nodes_pool,256],initializer = tf.truncated_normal_initializer(stddev = 0.1))
        if regularizer != None:
            tf.add_to_collection("loss",regularizer(fc1_weights))
        fc1_biasses = tf.get_variable("biasses",[256],initializer = tf.constant_initializer(0.01))
        fc1 = tf.tanh(tf.matmul(reshaped,fc1_weights) + fc1_biasses)
        #if train:fc1 = tf.nn.dropout(fc1,0.5) #适当避免过拟合问题

    with tf.variable_scope("layer6_fc2"):
        fc2_weights = tf.get_variable("weights",[256,3],initializer = tf.truncated_normal_initializer(stddev = 0.1))
        if regularizer != None:
            tf.add_to_collection("loss",regularizer(fc2_weights))
        fc2_biasses = tf.get_variable("biasses",[3],initializer = tf.constant_initializer(0.01))
        fc2 = tf.matmul(fc1,fc2_weights) + fc2_biasses

    return fc2
