#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import ipdb


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    y = tf.constant([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]])
    y_=tf.constant([[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 0.0, 1.0]])
    # ipdb.set_trace()
    ysoft = tf.nn.softmax(y)
    cross_entropy = -tf.reduce_sum(y_*tf.log(ysoft))
    
    cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    cross_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    
    print("step1:softmax result=", sess.run(ysoft))
    print("step2:cross_entropy result=", sess.run(cross_entropy))
    print("Function(softmax_cross_entropy_with_logits) result=", sess.run(cross_entropy2))
    print("cross_entropy_loss result=", sess.run(cross_entropy_loss))
    sess.close()