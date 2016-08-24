import numpy as np
import tensorflow as tf
import math
from sys import stderr

def DAF(trX,layers,batch_range,activation):

    """

    Parameters
    ----------

    trX: input data

    batch_range : size of minibatch

    layers = layers of network (first layer must have neurons as number of 
                              features of input data and final layers must have number 
                              of features as final dimension)

    activation: activation function   ("sigmoid" or "tanh")

    Returns
    -------

    dataset into transformed dimension using stacked denoising encoder.
    """
    cur_input=trX
    learning_rate=0.001
    n_layer=len(layers)
    
    for i in range(1,n_layer):
        print "layer",i
        node=layers[i-1]
        x = tf.placeholder(tf.float32, [None, layers[i-1]], name='x')
        w_e = tf.Variable(
            tf.random_uniform([node, layers[i]],
                              -1.0 / math.sqrt(node),
                              1.0 / math.sqrt(node)))
        b_e = tf.Variable(tf.zeros([layers[i]]))
        w_d=tf.transpose(w_e)
        b_d = tf.Variable(tf.zeros([node]))
        if(activation=="sigmoid"):
          z=tf.nn.sigmoid(tf.matmul(x, w_e) + b_e)
          y=tf.nn.sigmoid(tf.matmul(z, w_d) + b_d)
        elif(activation=="tanh"):
          z=tf.nn.tanh(tf.matmul(x, w_e) + b_e)
          y=tf.nn.tanh(tf.matmul(z, w_d) + b_d)
        else:
          print "Wrong Activation"
        cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
        l2_loss = tf.add_n([tf.nn.l2_loss(w_e),tf.nn.l2_loss(w_d)])
        loss = cost + 0.001*l2_loss
        train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)
        epoch=100
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for epoch_i in range(epoch):
                for start, end in zip(range(0, len(cur_input), batch_range), 
                                      range(batch_range, len(cur_input), batch_range)):
                    input_ = cur_input[start:end]
                    sess.run(train_op,feed_dict={x:input_})
                s= "\repoch: %d cost: %f"%(epoch_i,sess.run(cost,feed_dict={x:cur_input}))
                stderr.write(s)
                stderr.flush()
            cur_input=sess.run(z,feed_dict={x:cur_input})
    return cur_input