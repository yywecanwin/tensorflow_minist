# -*- coding: utf-8 -*-
# author：yaoyao time:2019/7/26
import numpy as np
import tensorflow as tf
from flask import Flask,jsonify,render_template,request
from mnist import model

x = tf.placeholder("float",[None,784])
sess = tf.Session()

#那去回归模型
with tf.variable_scope("regession"):
    y1,variables = model.regression(x)
saver = tf.train.Saver(variables)
#通过restore来获取回归模型
saver.restore(sess,"mnist/data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2,variables = model.convolutional(x,keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess,"mnist/data/convalutional.ckpt")

# 线性
def regression(input):
    return sess.run(y1,feed_dict={x:input}).flatten().tolist()

# 卷积
def convalutional(input):
    return sess.run(y2,feed_dict={x:input,keep_prob:1.0}).flatten().tolist()

app = Flask(__name__)

@app.route('/api/mnist',methods=['post'])
def mnist():
    input = ((255 - np.array(request.json,dtype=np.uint8))/255.0).reshape(1,784)
    output1 = regression(input)
    output2 = convalutional(input)
    return jsonify(results=[output1,output2])

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=8000)





