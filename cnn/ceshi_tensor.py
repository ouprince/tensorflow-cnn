import tensorflow as tf

with tf.Session() as sess:
    dd = tf.Variable([[0,0,1],[0,1,0],[1,0,0]],dtype = tf.float32)
    ss = tf.reshape(dd,[3,3])
    st = tf.nn.softmax_cross_entropy_with_logits(logits = ss,labels = dd)
    x = sess.run(st)
    print x
