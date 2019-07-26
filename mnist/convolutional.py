# -*- coding: utf-8 -*-
# author：yaoyao time:2019/7/24

import os
import model
import tensorflow as tf
import input_data
data = input_data.read_data_sets('MNIST_data',one_hot=True)
with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32,[None,784],name="x")
    keep_prob = tf.placeholder(tf.float32)
    y,variables = model.convolutional(x,keep_prob)

#训练
# 定义训练参数
y_ = tf.placeholder(tf.float32,[None,10],name='y')
#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#训练步长，随机梯度下降的方式
train_setp = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#预测，判断参数是否相等
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#保存
saver = tf.train.Saver(variables)

with tf.Session() as sess:
    #合并参数
    merged_summary_op = tf.summary.merge_all()
    #把参数的路径图
    summay_writer = tf.summary.FileWriter("/tmp/mnist_log/1",sess.graph)
    #添加图
    summay_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    #训练
    for i in range(20000):
        #训练大小
        batch = data.train.next_batch(50)
        #
        if i%100 == 0:
            #每个一百次，就打印一下准确率
            train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print("step %d, training accuracy %g " % (i,train_accuracy))
        sess.run(train_setp,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    #测试集
    print(sess.run(accuracy,feed_dict={x:data.test.images,y_:data.test.labels,keep_prob:1.0}))

    path = saver.save(
        sess,os.path.join(os.path.dirname(__file__),'data','convalutional.ckpt'),
        write_meta_graph = False,write_meta_state = False)

    #打印最后保存的路径
    print("Saved",path)
